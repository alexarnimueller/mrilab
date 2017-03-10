# -*- coding: utf-8 -*-
"""
.. module:: ml16.preprocessing

.. moduleauthor:: Alex Müller
"""
import os
from os.path import join

import nibabel as nib
import numpy as np


def dir_to_filelist(directory, ending=".nii"):
    """Function to read all file names with a given ending from a directory into a list.

    :param directory: {str} path of directory from which files should be read
    :param ending: {str} file ending to consider
    :return: array of file names in this directory
    """
    fnames = np.empty((1, 1), dtype='|S24')
    
    for f in os.listdir(directory):
        if f.endswith(ending):
            fnames = np.append(fnames, f)
        else:
            continue
    return fnames[1:]


def fnames_to_targets(directory, extension=".nii"):
    """Function to read all filenames from the given directory and connect them in a dictionary to the corresponding
    target values, which must be stored in a file called 'targets.csv' and located in the same directory. The file
    names must have the format "<something>_ID.<ending>" with ID corresponding to the row ID in the targets.csv file.

    :param directory: {str} path of directory from which files should be read
    :param extension: {str} file name extension to consider
    :return: a dictionary with file names as keys and targets as values
    """
    filenames = dir_to_filelist(directory, ending=extension)
    try:
        targets = np.genfromtxt(os.path.join(directory, 'targets.csv'))
        # connect file names to corresponding targets via the ID in the file names
        indices = [int(i.split('_')[1].split('.')[0]) - 1 for i in filenames]
        d = {str(f): int(targets[indices[n]]) for n, f in enumerate(filenames)}
        return d
    
    except IOError:
        print("\tNo file named 'targets.csv' in given directory! Target values set to 0.")
        d = {str(f): 0 for n, f in enumerate(filenames)}
        return d


def get_img_matrix(filename, slices=None):
    """Function to read a nifti img into numpy pixel vectors

    :param filename: {str} file name of the MRI img to read
    :param slices: {int} default: ``None`` (load all); whether the function should only load ``slices`` slices of
        the nifti image.
    :return: numpy array with all slices as pixel matrices
    """
    img = nib.load(filename)
    if slices:
        data = img.get_data()
        n = data.shape[2] / (slices + 1)  # first slice index
        sel = [n * (i + 1) for i in range(slices)]  # generate a list of slice indices
        data = np.array([data[..., i, 0] for i in sel])  # select these slices
    else:
        data = img.get_data()
    img.uncache()
    return data


def image_to_vector(img):
    """Function to flatten an img into a vector.

    :param img: {img array} image to be reshaped into array
    :return: {np.array} vectorized image
    """
    if len(img.shape) == 1:
        return np.array(img)
    else:
        s = img.shape[0] * img.shape[1]
        img_vect = img.reshape(1, s)
        return img_vect[0]


def slice_mri(img):
    """Function to get all horizontal slices of a nifit img, flatten them into vectors and store them in a matrix.

    :param img: {img matrix} nifti img as 3D pixel matrix
    :return: {np.array} matrix with pixel vectors of every slice
    """
    data = []
    if len(img.shape) == 4:  # slices in shape[2]
        for i in range(img.shape[2]):
            data.append(image_to_vector(img[..., i, 0]))
        return np.array(data).reshape((img.shape[0], img.shape[0] * img.shape[1]))
    elif len(img.shape) == 3:  # slices in shape[0]
        for i in range(img.shape[0]):
            data.append(image_to_vector(img[i, ...]))
        return np.array(data).reshape((img.shape[0], img.shape[1] * img.shape[2]))
    elif len(img.shape) == 2:
        return image_to_vector(img)


def files_to_data(file_dict, directory, num=None, slices=None):
    """Function to generate a data and target array from a dictionary containing file names (keys) and target
    values (values).

    :param file_dict: {dict} dictionary containing file names as keys and corresponding target values as values.
    :param directory: {str} directory from which files in ``file_dict`` should be loaded.
    :param num: {int} how many files should be read (random order, meant for testing purposes).
    :param slices: {int} default: ``None`` (load all); whether the function should only load ``slices`` slices of the nifti
        image.
    :return: two vectors: data array and target array.
    """
    # for every file: load img and put all slices as vectors in matrix
    x = []  # list to store image data
    y = []  # list to store target values
    print("\nLoaded:\n file name\t\tage")
    for i, (k, v) in enumerate(file_dict.items()[:num]):
        print(" %s \t%i" % (k, v))
        image = get_img_matrix(join(directory, k), slices=slices)
        m = slice_mri(image)
        t = np.array([v] * len(m))  # set target value for every slice
        x.extend(m)
        y.extend(t)
    
    if slices is None:
        slices = 176
    
    x = np.array(x).reshape(((i + 1) * slices, -1))
    y = np.array(y).reshape((x.shape[0], -1))[:, 0]
    return x, y


def get_standard_brain(x, num=278):
    """Evaluate every slice over all brains and return the "standard deviation brain" with stdevs as pixel values

    :param x: {numpy.array} array with every slice as linear vector of pixels, original shape (176,208)
    :param num: {int} number of brains in x_train
    :return: Brain array containing stdev values of all input brains in x_train and an array containing the important
        pixel indices for each slice.
    """
    x = x.reshape((num, 176, 36608))
    avrg_b = np.empty(shape=(176, 36608))
    std_b = np.empty(shape=(176, 36608))
    for i in range(176):
        avrg_b[i] = np.mean(x[:, i, :], axis=0)
        std_b[i] = np.std(x[:, i, :], axis=0)
    
    # get pixel indices which are over a given threshold (mean + std) per slice in the stdev brain
    index_b = dict()
    for s in range(176):
        bild = std_b[s]
        if np.mean(bild) != 0.0:
            index_b[s] = np.where(bild > (np.mean(bild) + np.std(bild)))[0]
    
    return avrg_b, std_b, index_b


def reduce_data_by_indexbrain(x, index_brain, y=None, num=278):
    """Extract the pixels most different in all brains (given in "indexbrain") from every slice of x_train.

    :param x: {numpy.array} training data to reduce
    :param y: {numpy.array} target values for training data
    :param index_brain: {numpy.array} output from :py:func:`get_standard_brain` with *important* pixel indices
    :param num: {int} number of brains in x_train (each brain has 176 slices)
    :return: reduced data x to important pixels.
    """
    x = x.reshape((num, 176, 36608))
    x_s = list()
    for b in range(num):
        slicearray = list()
        for k in index_brain.keys():
            slicearray.append(x[b, k, index_brain[k]])
        x_s.append(np.concatenate(slicearray))
    
    x_s = np.array(x_s).reshape((num, len(x_s[0])))
    y_s = []
    
    if y is not None:
        y_s = np.array([y[n * 176] for n in range(num)])
    
    return x_s, y_s
