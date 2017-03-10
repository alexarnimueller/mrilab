from nilearn.decoding import SpaceNetRegressor
import numpy as np
from time import time
from nilearn import datasets
from nilearn.plotting import plot_stat_map
from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
#from mrilab.brain import TrainingData
#from mrilab.preprocessing import fnames_to_targets

#test_dir = '/Users/modlab/x/documents/lectures/ml16/data/set_train/'
#d = fnames_to_targets(test_dir)
#brains = TrainingData(d, test_dir, num=50, random=True)
#brains.combine_brains()

n_subjects = 200  # increase this number if you have more RAM on your box
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
age = np.array(age)
gm_imgs = np.array(dataset_files.gray_matter_maps)

# Split data into training set and test set
rng = check_random_state(42)
gm_imgs_train, gm_imgs_test, age_train, age_test = train_test_split(
    gm_imgs, age, train_size=.6, random_state=rng)

# Sort test data for better visualization (trend, etc.)
perm = np.argsort(age_test)[::-1]
age_test = age_test[perm]
gm_imgs_test = gm_imgs_test[perm]

# To save time (because these are anat images with many voxels), we include
# only the 5-percent voxels most correlated with the age variable to fit.
# Also, we set memory_level=2 so that more of the intermediate computations
# are cached. Also, you may pass and n_jobs=<some_high_value> to the
# SpaceNetRegressor class, to take advantage of a multi-core system.
#
# Also, here we use a graph-net penalty but more beautiful results can be
# obtained using the TV-l1 penalty, at the expense of longer runtimes.
decoder = SpaceNetRegressor(memory="nilearn_cache", penalty="graph-net",
                            screening_percentile=5., memory_level=5, n_jobs=-1, verbose=0)
a = time()
decoder.fit(gm_imgs_train, age_train)  # fit
print "fitting took %s secs" % (time() - a)
coef_img = decoder.coef_img_
y_pred = decoder.predict(gm_imgs_test).ravel()  # predict
mse = np.mean(np.abs(age_test - y_pred))
print('Mean square error (MSE) on the predicted age: %.2f' % mse)

# Plot
# weights map
background_img = gm_imgs[0]
plot_stat_map(coef_img, background_img, title="graph-net weights",
              display_mode="z", cut_coords=1)
