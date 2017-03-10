"""
testing clustering with nilearn & scikit learn
"""
import time
from nilearn import input_data
from nilearn.image import mean_img
from sklearn.feature_extraction import image
from sklearn.cluster import FeatureAgglomeration
from nilearn.plotting import plot_roi, plot_epi, show


filename = 'tests/data/train_10.nii'

nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)

# The fit_transform call computes the mask and extracts the time-series
# from the files:
fmri_masked = nifti_masker.fit_transform(filename)

# We can retrieve the numpy array of the mask
mask = nifti_masker.mask_img_.get_data().astype(bool)

shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

start = time.time()
ward = FeatureAgglomeration(n_clusters=100, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
ward.fit(fmri_masked)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# Compute the ward with more clusters, should be faster as we are using
# the caching mechanism
#start = time.time()
#ward = FeatureAgglomeration(n_clusters=200, connectivity=connectivity,
#                            linkage='ward', memory='nilearn_cache')
#ward.fit(fmri_masked)
#print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

# Unmask the labels

# Avoid 0 label
labels = ward.labels_ + 1
labels_img = nifti_masker.inverse_transform(labels)

mean_func_img = mean_img(filename, n_jobs=-1)


first_plot = plot_roi(labels_img, mean_func_img, title="Ward parcellation",
                      display_mode='xz')

# common cut coordinates for all plots
cut_coords = first_plot.cut_coords

# Display the original data
plot_epi(nifti_masker.inverse_transform(fmri_masked[0]),
         cut_coords=cut_coords,
         title='Original (%i voxels)' % fmri_masked.shape[1],
         vmax=fmri_masked.max(), vmin=fmri_masked.min(),
         display_mode='xz')

# A reduced data can be created by taking the parcel-level average:
# Note that, as many objects in the scikit-learn, the ward object exposes
# a transform method that modifies input features. Here it reduces their
# dimension
fmri_reduced = ward.transform(fmri_masked)

# Display the corresponding data compressed using the parcellation
fmri_compressed = ward.inverse_transform(fmri_reduced)
compressed_img = nifti_masker.inverse_transform(fmri_compressed[0])

plot_epi(compressed_img, cut_coords=cut_coords,
         title='Compressed representation (2000 parcels)',
         vmax=fmri_masked.max(), vmin=fmri_masked.min(),
         display_mode='xz')

show()
