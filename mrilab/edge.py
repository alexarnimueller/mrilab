# -*- coding: utf-8 -*-
"""
Module with edge detection functions.

Available:
    - Sobel filter
    - Prewitt filter
    - Otsu threshold

.. module:: mrilab.edge

.. moduleauthor:: Alex Müller
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


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
    plt.imshow(img, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('Original', fontsize=20)
    plt.subplot(122)
    plt.imshow(filter, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('Filtered', fontsize=20)
    plt.show()
