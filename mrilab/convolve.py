# -*- coding: utf-8 -*-
"""
.. module:: ml16.convolve

.. moduleauthor:: Alex Müller
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg

# filters
horr_edge = np.array([[0,  0, -1,  0,  0],
                      [0,  0, -1,  0,  0],
                      [0,  0,  3,  0,  0],
                      [0,  0,  0,  0,  0],
                      [0,  0,  0,  0,  0]])

vert_edge = np.array([[0,  0, -1,  0,  0],
                      [0,  0, -1,  0,  0],
                      [0,  0,  6,  0,  0],
                      [0,  0, -1,  0,  0],
                      [0,  0, -1,  0,  0]])

diagl_edge = np.array([[-1,  0,  0,  0,  0],
                      [  0, -2,  0,  0,  0],
                      [  0,  0,  8,  0,  0],
                      [  0,  0,  0, -2,  0],
                      [  0,  0,  0,  0, -1]])

diagr_edge = np.array([[0,  0,  0,  0, -1],
                      [ 0,  0,  0, -2,  0],
                      [ 0,  0,  8,  0,  0],
                      [ 0, -2,  0,  0,  0],
                      [-1,  0,  0,  0,  0]])

emboss = np.array([[-1, -1, -1, -1,  0],
                   [-1, -1, -1,  0,  1],
                   [-1, -1,  0,  1,  1],
                   [-1,  0,  1,  1,  1],
                   [ 0,  1,  1,  1,  1]])

all_edge = np.array([[-2, -2, -2],
                     [-2,  8, -2],
                     [-2, -2, -2]])

sharp = np.array([[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]])

sharper = np.array([[1, 1, 1],
                    [1, 7, 1],
                    [1, 1, 1]])

filters = [horr_edge, vert_edge, diagl_edge, diagr_edge, emboss, all_edge, sharp, sharper]


def convolve_image(img, filters, plot=False):
    """Function to convolve an image with given filters and return the convoluted image.
    
    :param img: {numpy.array} image pixel matrix to convolve
    :param filters: {list} list of filter matrices to use for convolution
    :param plot: {bool} whether to plot the resulting convoluted images
    :return: {numpy.array} convoluted image
    """
    result = list()
    for f in filters:
        m = sg.convolve(img, f, "same")
        if plot:
            plt.imshow(m)
        result.extend(m)
    return np.array(result)
