import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from nilearn.image import load_img
from nilearn.masking import apply_mask
from nilearn.masking import compute_background_mask
from nilearn.plotting import plot_roi
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler


class Brain(object):
    """
    Base class for loading brain MRI images (nifti ``.nii`` format)
    """
    
    def __init__(self, filename):
        """
        :param filename: {str} filename of brain MRI image (.nii format) to read
        """
        self.img = load_img(filename)
        self.header = dict(self.img.header)
        self.name = os.path.split(filename)[1].split('.')[0]  # get name from file name
        self.shape = self.img.shape
        self.affine = self.img.affine
        self.data = np.zeros(shape=self.shape)
        self.loaded = False
        self.vectors = np.array([])
        self.mask = np.array([])
        self.mean = np.array([])
    
    def get_data(self):
        """Get the pixel matrix from the loaded image into a numpy array and store it in the attribute :py:attr:`data`.

        :return: pixel matrix in the attribute :py:attr:`data`
        """
        self.data = np.asarray(self.img.get_data())
        if len(self.shape) > 3:
            self.shape = self.shape[:3]
            self.data = self.data.reshape(self.shape)
        self.img.uncache()
        self.loaded = True
    
    def reshape_to_vector(self, axis=0, append=False):
        """Reshape the brain slices in the attribute :py:attr:`data` to 1-dimensional vectors and store them in the
        attribute :py:attr:`vectors`

        .. note:: The vecotrs get the maximal possible dimensionality according to the input data shape. If there are
            less pixels, the tail of the arrays is padded with zeros!

        :param axis: {int} axis to use for slice generation
        :param append: {bool} whether the produced vecotrs should be appended to the already existing ones
        :return: pixel data as 1-dimensional array per slice in the attribute :py:attr:`data`
        """
        if not self.loaded:
            self.get_data()
            self.loaded = True
        m = list()
        shapelist = sorted(list(self.shape), reverse=True)
        maxshape = shapelist[0] * shapelist[1]
        for i in range(self.shape[axis]):
            l = np.zeros(maxshape)  # used for zero padding of smaller vertical slices
            if axis == 0:
                s = self.shape[1] * self.shape[2]
                l[:s] = self.data[i, :, :].reshape(s)
                m.append(l)
            elif axis == 1:
                s = self.shape[0] * self.shape[2]
                l[:s] = self.data[:, i, :].reshape(s)
                m.append(l)
            elif axis == 2:
                s = self.shape[0] * self.shape[1]
                l[:s] = self.data[:, :, i].reshape(s)
                m.append(l)
            else:
                print("Unknown axis dimension!")
                break
        if append:
            if len(self.vectors) > 0:  # check if array is still empty, then set it to the first slice
                self.vectors = np.array(m)
            else:
                self.vectors = np.append(self.vectors, np.array(m), axis=0)
        else:
            self.vectors = np.array(m)
    
    def inspect_slices(self, filename=None):
        """Method to plot the center slices of the brain in all dimensions to gray-scale images.

        :param filename: {str} filename if plot should be saved to file
        :return: plot with 3 brain slices
        """
        if not self.loaded:
            self.get_data()
            self.loaded = True
        x = int(self.shape[0] / 2)
        y = int(self.shape[1] / 2)
        z = int(self.shape[2] / 2)
        s_x = ndimage.rotate(self.data[x, :, :], 90)
        s_y = ndimage.rotate(self.data[:, y, :], 90)
        s_z = self.data[:, :, z]
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        for a, b in zip(ax, [s_x, s_y, s_z]):
            a.imshow(b, cmap='Greys_r')
            a.set_axis_off()
        fig.suptitle('3 brain slices from the middle of each orientation.', fontweight='bold', fontsize=18)
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
    
    def scale_data(self, srange=(0, 255)):
        """Method to scale the given MRI image (whole image data) in the attribute :py:attr:`data` to grayscale (0,
        255).

        :param srange: {tuple} range used for scaling
        :return: scaled image in :py:attr:`data`
        """
        if not self.loaded:
            self.get_data()
            self.loaded = True
        sclr = MinMaxScaler(srange)
        tmp = self.data.reshape((1, -1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_s = sclr.fit_transform(tmp.T).T
        self.data = tmp_s.reshape(self.shape)
    
    def mask_background(self, apply=False):
        """Method to generate a mask matrix which does not consider blank pixels outside the brain area.

        :param apply: {bool} whether the mask should directly be applied to extract the wanted pixels.
        :return: mask as boolean ``numpy.array`` in the attribute :py:attr:`mask`. If ``apply=True``, data in the
            attribute :py:attr:`data` is reshaped as well.
        """
        self.mask = compute_background_mask(self.img)
        if apply:
            self.data = apply_mask(self.img, self.mask)
    
    def inspect_mask(self):
        """Method to plot the computed mask.

        :return: three middle slices with the applied mask on the mean image.
        """
        try:
            plot_roi(self.mask, self.img)
        except TypeError:
            raise LookupError("\nNo mask precomputed! First compute a mask before applying this method.")
