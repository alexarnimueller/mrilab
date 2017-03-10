README
======

**Machine Learning Project 1**

Group: **mrilab**

:author: `Alex Müller <alex.mueller@pharma.ethz.ch>`_, `Erik Gawehn <erik.gawehn@pharma.ethz.ch>`_

This Python package can be used for predicting patient age from given MRI images. The module :py:mod:`brain` contains
the class :class:`MultiBrain` which reads Niftii MRI images into one single object, stacking the individual brains in
the 4th dimension. The module :py:mod:`preprocessing` adds some simplification to read file names, modules
:py:mod:`convolve` and :py:mod:`edge` can be used for image processing. The module :py:mod:`gridsearch` gives the
possibility to search the best parameters for a specified model.

Workflow
--------
The training and prediction workflow is located in :file:`predict_final.py`.

A brief explanation of the steps in the applied workflow:

1) loading data from Nifti image files and combine them into one :class:`MultiBrain` instance
2) averaging over 2x2x2 voxel boxes for 8-fold dimensionality reduction
3) calculating the correlation of every voxel to the age of the brain
4) picking only voxels that have a higher absolute correlation than mean + 2 * std
5) training of a pipeline with:
5.1) Standard scaler (zero mean and unit variance)
5.2) Linear support vector regressor with C=2. regularization
6) predicting ages of test images with trained pipeline

Reasoning
---------
The main objective of this task turned out to be dimensionality reduction while keeping the most informative voxels. We
tackled this problem by smoothing the images through averaging over small boxes of 8 voxels first, followed by
calculation of the correlation of these averaged resulting voxels with the brain age. We were then able to select the
most correlated brain voxels for training with an arbitrary threshold of correlations over ``mean + 2 * std``. These
voxels were then standard scaled to zero mean and unit variance before being fed to a linear Support Vector Regressor
(SVR). The C parameter of the SVR was obtained through a grid search workflow in :file:`gridsearch.py`.
