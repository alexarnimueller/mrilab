# -*- coding: utf-8 -*-
"""
Final training and prediction workflow for MLP3 Group **mrilab**

.. author:: Alex Müller, Erik Gawehn
"""

import os
import warnings
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from os.path import join, abspath
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from nilearn.image import load_img, smooth_img, mean_img, index_img
from nilearn.masking import apply_mask, compute_epi_mask, unmask
from nilearn.masking import compute_background_mask
from nilearn.plotting import plot_roi, plot_img
from scipy import ndimage
from skimage.draw import ellipsoid
from skimage.filters import threshold_otsu
from skimage.transform import downscale_local_mean
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, hamming_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def reader3(directory, ending=".nii", dtype='train', startid=1, endid=278):
    filenames = [os.path.join(directory, dtype + '_' + str(id) + ending) for id in range(startid, endid + 1)]
    try:
        targets = np.genfromtxt(os.path.join(directory, 'targets.csv'), delimiter=',')
        d = OrderedDict({})
        for i, f in enumerate(filenames):
            d[f] = targets[i]
        return d
    except IOError:
        print("\tNo file named 'targets.csv' in given directory %s!\n\tTarget values set to 0." % directory)
        d = OrderedDict({})
        for i, f in enumerate(filenames):
            d[f] = 0.
        return d


def sobel(img):
    """Function to apply a sobel filter on a given input image.

    :param img: {numpy.array} image as numpy array
    :return: {numpy.array} filtered image
    """
    sx = ndimage.sobel(img, axis=0, mode='constant')
    sy = ndimage.sobel(img, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob


def prewitt(img):
    """Function to apply a prewitt filter on a given input image.

    :param img: {numpy.array} image as numpy array
    :return: {numpy.array} filtered image
    """
    sx = ndimage.prewitt(img, axis=0, mode='constant')
    sy = ndimage.prewitt(img, axis=1, mode='constant')
    prew = np.hypot(sx, sy)
    return prew


def otsu(img):
    """Function to apply Otsu thresholding on a given image.

    :param img: {numpy.array} image to apply threshold on
    :return: {numpy.array} filtered image
    """
    val = threshold_otsu(img)
    o = img < val
    return o, val


def plot_filter(img, filter):
    """Function to plot a given image and the generated sobel filters

    :param img: {numpy.array} original image
    :param filter: {numpy.array} filtered image
    :return: plot of original and filtered image
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original', fontsize=20)
    plt.subplot(122)
    plt.imshow(filter)
    plt.axis('off')
    plt.title('Filtered', fontsize=20)
    plt.show()


def _cov_comp(vec1, vec2):
    """Private function for parallel covariance calculation"""
    covmat = np.cov(vec1, vec2)
    return covmat[0, 1]


def _cor_comp(vec1, vec2):
    """Private function for parallel correlation calculation"""
    cormat = np.corrcoef(vec1, vec2)
    return cormat[0, 1]


def _vox_comp(tup, img):
    """Private function for parallel down-voxeling"""
    return downscale_local_mean(img, tup)


def _subsample_cube(cubesize, style, img):
    """Private function for parallel down-cube-downsampling"""
    img = img.T
    cubes = []
    for x in range(0, img.shape[0], cubesize):
        for y in range(0, img.shape[1], cubesize):
            for z in range(0, img.shape[2], cubesize):
                cube = img[x:x + cubesize, y:y + cubesize, z:z + cubesize]
                if style == 'max':
                    cubes.append(np.max(cube))
                elif style == 'mean':
                    cubes.append(np.mean(cube))
                elif style == 'hist':
                    cubes.extend(np.histogram(cube, 5)[0])
    return cubes


def _rescale(bins, max, img):
    """Private function for parallel rescaling"""
    img *= float(bins - 1) / float(max)
    return img


def _smooth(fwhm, img):
    """Private function for parallel image smoothing"""
    return smooth_img(img, fwhm)


class MultiBrain(object):
    """
    Class for combining many brain MRI images into data and target arrays for training a model.
    """
    
    def __init__(self, filename_dict, directory, num=None, sort=True, sparse=False):
        """
        :param filename_dict: {dict} Dictionary containing file names as keys and ages as values
        :param directory: {str} path of the folder containing all files given in ``filename_dict``
        :param num: {int} number of files to select (for test purposes)
        :param sort: {bool} whether the images should be selected according to their location in the read dictionary
            (``False``) or sorted according to the sorted target values (``True``).
        :param sparse: {bool} whether only one image per target value should be loaded.
        """
        if sparse:  # then only take one example per value
            l_sorted = dict()
            for k, v in filename_dict.items():
                if v not in l_sorted.values():  # if value not yet present in l_sorted, add the pair
                    l_sorted[k] = v
            if sort:
                l_sorted = sorted(l_sorted.items(), key=lambda x: x[1])[:num]  # sorted list according to target values
            else:
                l_sorted = np.random.choice(np.array(l_sorted.items()), size=num, replace=False)
        elif sort:
            l_sorted = sorted(filename_dict.items(), key=lambda x: x[1])[:num]  # sorted list according to target values
        else:
            l_sorted = filename_dict.items()[:num]
        
        self.img = None
        self.mask = None
        self.data = None
        self.mean = None
        self.std = None
        self.cov = None
        self.cor = None
        self.scaler = None
        self.vox = None
        self.is_smoothed = False
        self.directory = directory
        self.targets = np.array([x[1] for x in l_sorted])  # ascending target values
        self.names = np.array([x[0] for x in l_sorted])  # sorted file names according to target values
        self.filenames = np.array([join(directory, f) for f in self.names])
    
    def combine_brains(self, slices=None):
        """Method to combine all brains in the loaded filename dictionary into a big data object with the individual
        brain data in the 4th dimension.

        :param slices: {int} number of slices to load from each brain (testing purpose). The average from ±1 slice
            is loaded for every slice.
        :return: data of all files loaded into the attribute :py:attr:`data`
        """
        print("Loading %i files..." % len(self.names))
        self.img = load_img(self.filenames)
        if slices:
            step = int(float(self.img.shape[2]) / float(slices + 1))
            newshape = list(self.img.shape)
            newshape[2] = slices
            imgarr = np.empty(shape=tuple(newshape))
            for s, img in enumerate(self.img.dataobj.T):
                for i in range(1, slices + 1):
                    imgarr[..., i - 1, s] = np.mean(img.T[..., (i * step - 1):(i * step + 1)], axis=2)
            self.img = Nifti1Image(imgarr.reshape((self.img.shape[0], self.img.shape[1], slices, len(self.filenames))),
                                   self.img.affine)
        self.img.uncache()
        print("\tAll files loaded!")
    
    def stack_all_slices(self):
        """Method to combine all brain slices of all brains in 4D into one big array in the attribute :py:attr:`data`.
        The target values in the attribute :py:attr:`targets` is updated to match target values for every slice.

        :return: stacked slices in :py:attr:`data` and corresponding target values in :py:attr:`targets`.
        """
        print("Concatenating slices...")
        self.data = np.concatenate(self.img.dataobj.T).T
        self.img.uncache()
        self.targets = np.array([[t] * self.img.shape[2] for t in self.targets]).reshape((1, -1)).tolist()[0]
        print("\tSlices concatenated!")
    
    def smooth(self, fwhm=4, numproc=4):
        """Method to smooth the brains with a Gaussian filter.

        :param fwhm: {number} Smoothing strength, as a Full-Width at Half Maximum, in millimeters.
        :param numproc: {int} number of parallel processes applied to smooth the brain.
        :return: smoothed image in the attribute :py:attr:`img`
        """
        print("Smoothing...")
        p = Pool(numproc)  # number of parallel processes
        try:
            self.img = load_img(p.map(partial(_smooth, fwhm), [index_img(self.img, n) for n in range(self.img.shape[
                                                                                                         -1])]))
        finally:
            p.close()
            p.join()
        self.img.uncache()
        self.is_smoothed = True
        print("\tSmoothed!")
    
    def background_mask(self, calculate=True, apply=False, plot=True):
        """Compute the average background mask for all given brains.

        :param calculate: {bool} do calculate or just use other options.
        :param apply: {bool} if True, the mask is directly applied to cut down the brains in :py:attr:`data` and
            reshape them into vectors
        :param plot: {bool} whether the generated mask should be visualized in a plot. The first image in 4th
            dimension is shown.
        :return: mask matrix in the attribute :py:attr:`mask` and if `apply=True` data vectors in :py:attr:`data`.
        """
        print("Computing background mask...")
        if calculate:
            self.mask = compute_background_mask(self.img)
        if apply:
            self.apply_mask()
        if plot:
            plot_roi(self.mask, Nifti1Image(self.img.dataobj[..., 0], self.img.affine))
        print("\tBackground mask computed!")
    
    def epi_mask(self, calculate=True, apply=False, plot=True):
        """Compute the average background mask for all given brains.

        :param calculate: {bool} do calculate or just use other options.
        :param apply: {bool} if True, the mask is directly applied to cut down the brains in :py:attr:`data` and
            reshape them into vectors
        :param plot: {bool} whether the generated mask should be visualized in a plot. The first image in 4th
            dimension is shown.
        :return: mask matrix in the attribute :py:attr:`mask` and if `apply=True` data vectors in :py:attr:`data`.
        """
        print("Computing EPI mask...")
        if calculate:
            self.mask = compute_epi_mask(self.img)
        if apply:
            self.apply_mask()
        if plot:
            plot_roi(self.mask, Nifti1Image(self.img.dataobj[..., 0], self.img.affine))
        print("\tEPI mask computed!")
    
    def mean_brain(self, std=True, plot=True):
        """Method to generate an average brain from all brain volumes in the 4th dimension.

        :param std: {bool} whether the standard deviation of all brain volumes should be computed
        :param plot: {bool} whether the generated brains should be visualized
        :return: depending on the given options an averaged brain and / or standard-deviation-brain as numpy.array
        """
        print("Computing mean brain...")
        self.mean = mean_img(self.img, n_jobs=-1)
        if plot:
            plot_img(self.mean, title='Averaged Brain')
        print("\tMean brain computed!")
        if std:
            self.std_brain(plot=plot)
    
    def std_brain(self, plot=True, mask=False):
        """Method to generate an brain of standard deviations from all brain volumes in the 4th dimension.

        :param plot: {bool} whether the generated brains should be visualized
        :param mask: {bool} whether the mask in the attribute :py:attr:`mask` should be applied to the image before.
        :return: depending on the given options an averaged brain and / or standard-deviation-brain as numpy.array
        """
        print("Computing std brain...")
        if mask:
            if self.data is not None:
                vectorized = self.data
            else:
                vectorized = apply_mask(self.img, self.mask)
        else:
            vectorized = apply_mask(self.img, Nifti1Image(np.ones(self.img.shape[:3]), self.img.affine))
        
        self.std = Nifti1Image(np.std(vectorized, axis=0).reshape(self.img.shape[:3]),
                               self.img.affine)
        if plot:
            plot_img(self.std, title='Standard Deviation Brain')
        del vectorized
        self.std.uncache()
        print("\tStd brain computed!")
    
    def vox_brain(self, scaletuple=(4, 4, 4), plot=True, numproc=4):
        """Method to downsample a brain by averaging over a specified number of voxels

        :param scaletuple: {tuple} size of voxel box to average over
        :param plot: {bool} whether the generated brains should be visualized
        :param numproc: {int} number of parallel processes applied to calculate the covbrain.
        :return: the correlation brain as 3D Nifti1Image
        """
        print("Computing vox brain...")
        p = Pool(numproc)  # number of parallel processes
        try:
            self.data = np.array(p.map(partial(_vox_comp, scaletuple), self.img.dataobj.T)).T
        finally:
            p.close()
            p.join()
        self.data_to_img(shape=self.data.shape[:3])
        self.data = None
        
        if plot:
            plot_img(index_img(self.img, 0), title="Voxel brain")
        print("\tVox brain computed!\n\tNew shape: %s" % str(self.img.shape))
    
    def subsample_cubes(self, cubesize=5, style='max', numproc=12):
        """Method to cut down whole brain into small cubes of size *cubesize*. The resulting cubes can then be
        analysed by different styles: ['max', 'mean', 'hist']
        
        :param cubesize: {int} cube side length
        :param style: {str} available: ['max', 'mean', 'hist']
        :param numproc: {int} number of parallel processes to be used for calculation
        :return: cut down and transformed cube data in in :py:attr:`data`
        """
        print("Subsampling brain with cubes..")
        p = Pool(numproc)  # number of parallel processes
        try:
            self.data = np.array(p.map(partial(_subsample_cube, cubesize, style), self.img.dataobj.T))
        finally:
            p.close()
            p.join()
        print("\tSubsampling completed!")
    
    def cube_mask(self, x1=50, x2=120, y1=50, y2=150, z1=50, z2=100, apply=False, plot=True):
        """Method to cut out a cube from the center of the brain.

        :param x1: {int} start range x dimension
        :param x2: {int} stop range x dimension
        :param y1: {int} start range y dimension
        :param y2: {int} stop range y dimension
        :param z1: {int} start range z dimension
        :param z2: {int} stop range z dimension
        :param apply: {bool} if ``True``, the mask is directly applied to cut down the brains in :py:attr:`data` and
            reshape them into vectors
        :param plot: {bool} whether the generated mask should be visualized in a plot. A random image is shown.
        :return: Mask in the attribute :py:attr:`mask` and cut out cube in :py:attr:`data` if ``apply=True``.
        """
        print("Computing cube mask...")
        self.mask = np.zeros(self.img.shape[:3])
        self.mask[x1:x2, y1:y2, z1:z2] = 1
        self.mask = Nifti1Image(self.mask, self.img.affine)
        
        if apply:
            self.apply_mask()
        
        if plot:
            plot_roi(self.mask, index_img(self.img, np.random.randint(0, self.img.shape[-1])))
        print("\tMask computed!")
    
    def ellipsoid_mask(self, a=40, b=60, c=40, apply=False, plot=True):
        """Method to generate an ellipsoid to mask voxels in the brain.

        :param a: {int} Length of semimajor axis aligned with x-axis.
        :param b: {int} Length of semimajor axis aligned with y-axis.
        :param c: {int} Length of semimajor axis aligned with z-axis.
        :param apply: {bool} if ``True``, the mask is directly applied to cut down the brains in :py:attr:`data` and
            reshape them into vectors
        :param plot: {bool} whether the generated mask should be visualized in a plot. A random image is shown.
        :return: Mask in the attribute :py:attr:`mask`
        """
        print("Computing ellipsoid mask...")
        ellps = ellipsoid(a, b, c)
        # shapes
        q, r, s = ellps.shape
        x, y, z = self.img.shape[:3]
        
        # get spacing to walls for centering the ellipsoid in the middle of the brain
        a = (x - q) / 2
        c = (y - r) / 2
        e = (z - s) / 2
        if (x - q) % 2:
            b = a + 1
        else:
            b = a
        if (y - r) % 2:
            d = c + 1
        else:
            d = c
        if (z - s) % 2:
            f = e + 1
        else:
            f = e
        
        ellps = np.pad(ellps, ((a, b), (c, d), (e, f)), mode='constant', constant_values=False).astype('int')
        self.mask = Nifti1Image(ellps, self.img.affine)
        
        if apply:
            self.apply_mask()
        
        if plot:
            plot_roi(self.mask, index_img(self.img, np.random.randint(0, self.img.shape[-1])))
        print("\tMask computed!")
    
    def cor_brain(self, numproc=4, plot=True, mask=True):
        """Method to generate an brain of correlations between all brain volumes in the 4th dimension and their labels.

        :param numproc: {int} number of parallel processes applied to calculate the covbrain.
        :param plot: {bool} whether the generated brains should be visualized
        :param mask: {bool} whether the mask in the attribute :py:attr:`mask` should be applied to the image before.
        :return: the correlation brain as 3D Nifti1Image
        """
        print("Computing correlation brain...")
        if mask:
            if self.data is not None:
                vectorized = self.data
            else:
                vectorized = apply_mask(self.img, self.mask)
        else:
            vectorized = apply_mask(self.img, Nifti1Image(np.ones(self.img.shape[:3]), self.img.affine))
        with warnings.catch_warnings():  # ignore RuntimeWarning
            warnings.simplefilter("ignore")
            if len(train.targets.shape) > 1:
                targets = self.targets[:, 0]
            else:
                targets = self.targets
            p = Pool(numproc)  # number of parallel processes
            try:
                corval = np.array(p.map(partial(_cor_comp, targets), vectorized.T))
            finally:
                p.close()
                p.join()
        corval = np.nan_to_num(corval)  # replace nan with 0
        if mask:
            self.cor = corval
        else:
            self.cor = Nifti1Image(corval.reshape(self.img.shape[:3]), self.img.affine)
            self.cor.uncache()
        if plot and not mask:
            plot_img(self.cor, title="Correlation brain")
        del vectorized
        print("\tCorrelation brain computed!")
    
    def cov_brain(self, numproc=4, plot=True, mask=True):
        """Method to generate an brain of covariances between all brain volumes in the 4th dimension and their labels.

        :param numproc: {int} number of parallel processes applied to calculate the covbrain.
        :param plot: {bool} whether the generated brains should be visualized
        :param mask: {bool} whether the mask in the attribute :py:attr:`mask` should be applied to the image before.
        :return: the covariance brain as 3D Nifti1Image
        """
        print("Computing covariance brain...")
        if mask:
            if self.data is not None:
                vectorized = self.data
            else:
                vectorized = apply_mask(self.img, self.mask)
        else:
            vectorized = apply_mask(self.img, Nifti1Image(np.ones(self.img.shape[:3]), self.img.affine))
        if len(train.targets.shape) > 1:
            targets = self.targets[:, 0]
        else:
            targets = self.targets
        p = Pool(numproc)  # number of parallel processes
        try:
            covval = np.array(p.map(partial(_cov_comp, targets), vectorized.T))
        finally:
            p.close()
            p.join()
        covval = np.nan_to_num(covval)  # replace nan with 0
        if mask:
            self.cov = covval
        else:
            self.cov = Nifti1Image(covval.reshape(self.img.shape[:3]), self.img.affine)
            self.cov.uncache()
        if plot and not mask:
            plot_img(self.cov, title="Covariance brain")
        del vectorized
        print("\tCovariance brain computed!")
    
    def apply_mask(self, mask=None, on_data=False):
        """Method to apply a precomputed mask in the attribute :py:attr:`mask` to the image.

        :param mask: {numpy.matrix} the mask to apply on the image. If ``None``, the mask in the attribute
            :py:attr:`mask` is used (if pre-calculated, e.g. with :py:func:`background_mask`)
        :param on_data: {bool} whether the mask should be applied on the data in the attribute :py:attr:`data`.
        :return: data vectors in :py:attr:`data`
        """
        print("Applying mask...")
        if mask is not None:
            if on_data:
                pass
            else:
                if type(mask) != Nifti1Image:
                    mask = Nifti1Image(mask, self.img.affine)
                self.data = apply_mask(self.img, mask)
        else:
            if type(self.mask) != Nifti1Image:
                self.mask = Nifti1Image(self.mask, self.img.affine)
            self.data = apply_mask(self.img, self.mask)
        
        self.img.uncache()
        print("\tMask applied!")
    
    def data_to_img(self, shape=(176, 208, 176)):
        """Method to transform the pixel arrays stored in the attribute :py:attr:`data` into a Nifti image object
        and overwrite the existing one in the attribute :py:attr:`img`.

        :param shape: {tuple} shape of the image that is generated
        :return: updated attribute :py:attr:`img` and cleared :py:attr:`data`
        """
        print("Transforming data array to image...")
        if len(self.data.shape) == len(self.img.shape):  # if data already has the right shape style
            self.img = Nifti1Image(self.data, self.img.affine)
        else:
            shape = tuple(list(shape) + [self.data.shape[0]])
            self.img = Nifti1Image(self.data.T.reshape(shape), self.img.affine)
        self.data = None
        print("\tTransformed!")
    
    def sobel_filter(self, plot=False, overwrite=True):
        """Method for applying a sobel filter to detect edges.

        :param plot: {bool} if ``True``, an some example results are plotted
        :param overwrite: {bool} whether the original image in the attribute :py:attr:`img` should be overwritten.
        :return: Detected edges in the attribute :py:attr:`data`
        """
        print("Applying sobel filter...")
        self.data = sobel(self.img.dataobj)
        if plot:
            plot_filter(self.img.dataobj[..., self.img.shape[2] // 2, 0], self.data[..., self.img.shape[2] // 2, 0])
        if overwrite:
            self.img = Nifti1Image(self.data, self.img.affine)
            self.data = None
        self.img.uncache()
        print("\tFiltered!")
    
    def prewitt_filter(self, plot=False, overwrite=True):
        """Method for applying a prewitt filter to detect edges.

        :param plot: {bool} if ``True``, an some example results are plotted
        :param overwrite: {bool} whether the original image in the attribute :py:attr:`img` should be overwritten.
        :return: Detected edges in the attribute :py:attr:`data`
        """
        print("Applying Prewitt filter...")
        self.data = prewitt(self.img.dataobj)
        if plot:
            plot_filter(self.img.dataobj[..., 90, 0], self.data[..., 90, 0])
        if overwrite:
            self.img = Nifti1Image(self.data, self.img.affine)
            self.data = None
        self.img.uncache()
        print("\tFiltered!")
    
    def rescale(self, nbins=255, numproc=4, overwrite=True):
        """Method to convert the brain images from MRI signals to a given pixel value scale

        :param nbins: {int} integer representing the number of bins generated.
        :param numproc: {int} number of parallel processes applied to rescale.
        :param overwrite: {bool} whether the original image in the attribute :py:attr:`img` should be overwritten.
        :return:
        """
        print("Rescaling pixel intensity range...")
        p = Pool(numproc)  # number of parallel processes
        try:
            self.data = np.array(p.map(partial(_rescale, nbins, np.max(self.img.dataobj)), self.img.dataobj.T.astype(
                    'float64'))).T
        finally:
            p.close()
            p.join()
        if overwrite:
            self.img = Nifti1Image(self.data, self.img.affine)
            self.data = None
        self.img.uncache()
        print("\tRescaled!")
    
    def standardscale(self, overwrite=True, fit=True, data=False):
        """Method to standard scale all pixel values to zero mean and 1 standard deviation.

        :param overwrite: {bool} whether the original image in the attribute :py:attr:`img` should be overwritten.
        :param fit: {bool} if ``Ture``, a new standard scaler is generated and fitted, if ``False``, the scaler in the
            attribute :py:attr:`scaler` is used for scaling
        :param data: {bool} whether to take the voxels in the attribute :py:attr:`data` as input.
        :return: scaled voxels in :py:attr:`data` or :py:attr:`img` (if ``overwrite=True``)
        """
        print("Rescaling pixel intensity range...")
        if data:
            vectorized = self.data
        else:
            vectorized = apply_mask(self.img, Nifti1Image(np.ones(self.img.shape[:3]), self.img.affine))
        if fit:  # instantiate new scaler, fit and transform
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(vectorized.T).T
        else:  # take trained scaler in self.scaler and transform
            self.data = self.scaler.transform(vectorized.T).T
        
        if overwrite:
            self.img = unmask(self.data, Nifti1Image(np.ones(self.img.shape[:3]), self.img.affine))
            self.data = None
            self.img.uncache()
        print("\tRescaled!")
    
    def plot_compare(self, num):
        """Plot the middle slices of different brains with their corresponding classes.

        :param num: {int} number of brains to plot per class
        :return: plot with ``2 * num`` subplots of the brain middle slices.
        """
        fig, ax = plt.subplots(num, 2, figsize=(10, 12))
        pos = index_img(self.img, np.where(np.array(self.targets) == 1)[0])  # all images with target = 1
        neg = index_img(self.img, np.where(np.array(self.targets) == 0)[0])  # all images with target = 0
        ax[0, 0].set_title('Class 1', fontweight='bold', fontsize=16)
        ax[0, 1].set_title('Class 0', fontweight='bold', fontsize=16)
        for n in range(num):
            ax[n, 0].imshow(pos.dataobj[..., self.img.shape[2] // 2, n])
            ax[n, 1].imshow(neg.dataobj[..., self.img.shape[2] // 2, n])
            for i in range(2):
                ax[n, i].set_xticklabels([])
                ax[n, i].set_yticklabels([])
                ax[n, i].set_xticks([])
                ax[n, i].set_yticks([])
        plt.tight_layout()


if __name__ == "__main__":
    np.random.seed(2881749)
    globtime = time()
    
    # get location of data and define output file location in this directory
    outfile = join(abspath('.'), 'predictions.csv')
    dir_train = join(abspath('.'), 'data/set_train')
    dir_test = join(abspath('.'), 'data/set_test')
    
    # read file names and combine brain images into one 4D brain
    datadict_train = reader3(dir_train, dtype='train', startid=1, endid=278)  # combining filenames with target values
    datadict_test = reader3(dir_test, dtype='test', startid=1, endid=138)
    train = MultiBrain(datadict_train, dir_train, sort=False)  # instantiate MultiBrain class with all needed methods
    test = MultiBrain(datadict_test, dir_test)
    train.combine_brains()  # appending all images in one Nifti object
    test.combine_brains()
    
    print "Preprocessing: subsample with cubes and make histograms..."
    train.subsample_cubes(cubesize=10, style='hist')
    test.subsample_cubes(cubesize=10, style='hist')
    
    print "Data dimension before pipeline:"
    print train.data.shape[1]

    # ##################### Models #####################
    # --------------------- Gender ---------------------
    print "Training gender pipeline..."
    pipeline = Pipeline([('var', VarianceThreshold()),
                         ('scl', StandardScaler()),
                         ('clf', LogisticRegression(n_jobs=-1, class_weight={1.: 1., 0.: 2.}))])
    
    pipeline.fit(train.data, train.targets[:, 0])
    
    print "Predicting test data for gender..."
    preds_gend = pipeline.predict(test.data)
    
    print "5-fold cross-validation..."
    scores = cross_val_score(pipeline, train.data, train.targets[:, 0], cv=5, n_jobs=-1, scoring=make_scorer(hamming_loss))
    print("Average gender hamming loss:  %.3f" % np.mean(scores))
    
    # --------------------- Age ---------------------
    print "Training age pipeline..."
    pipeline = Pipeline([('var', VarianceThreshold()),
                         ('scl', StandardScaler()),
                         ('clf', LogisticRegression(n_jobs=-1, class_weight={1.: 1., 0.: 1.5}))])
    
    pipeline.fit(train.data, train.targets[:, 1])
    
    print "Predicting test data for age..."
    preds_age = pipeline.predict(test.data)
    
    print "5-fold cross-validation..."
    scores = cross_val_score(pipeline, train.data, train.targets[:, 1], cv=5, n_jobs=-1, scoring=make_scorer(hamming_loss))
    print("Average age hamming loss:  %.3f" % np.mean(scores))
    
    # --------------------- Health ---------------------
    print "Training health pipeline..."
    pipeline = Pipeline([('var', VarianceThreshold()),
                         ('scl', StandardScaler()),
                         ('clf', LogisticRegression(n_jobs=-1, class_weight={1.: 1., 0.: 3.}))])
    
    pipeline.fit(train.data, train.targets[:, 2])
    
    print "Predicting test data for health..."
    preds_health = pipeline.predict(test.data)
    
    print "5-fold cross-validation..."
    scores = cross_val_score(pipeline, train.data, train.targets[:, 2], cv=5, n_jobs=-1, scoring=make_scorer(hamming_loss))
    print("Average health hamming loss:  %.3f" % np.mean(scores))
    
    # ##################### Saving Output #####################
    print "Saving predictions..."
    preds = np.vstack((preds_gend, preds_age, preds_health)).T
    samples = [int(s.split('_')[2].split('.')[0]) - 1 for s in test.names]  # get the IDs from the file names
    samples = np.array([preds.shape[1] * [s] for s in samples]).reshape((1, -1)).squeeze()
    labels = ['gender', 'age', 'health'] * preds.shape[0]
    preds_bool = preds.astype('bool').reshape((1, -1)).squeeze()
    df = pd.DataFrame(np.vstack((samples, labels, preds_bool)).T, columns=['Sample', 'Label', 'Predicted'])
    df.index.name = 'ID'
    df.to_csv(outfile)
    print "\nSaved predictions to %s" % outfile
    print("whole duration: %s" % (time() - globtime))
    print "\nDONE!"
