import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import metrics
from sklearn.decomposition import PCA


#### DATA PREP ####
# smaller number for faster convergence
train_samples = 5000
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#print(X.shape, y.shape)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=1000)

X_train = X_train/255
X_test = X_test/255

data = X_train
n_samples, n_features = data.shape
n_digits = len(np.unique(y_train))
labels = y_train

sample_size = 5000
# original dataset dimensionality
print("\n\nn_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

# Visualize the results on PCA-reduced data
# reduce from 784 to 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)
# reduced dimensionality
n_samples, n_features = reduced_data.shape
print("Reduced Dimensionality: \nn_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


###   code below this computes the K means and plots the output in 2d  ###
###   just use the above code to get the dimension reduction if you want to use your kmeans program ###
# #############################################################################



k_means = KMeans(init='k-means++', n_clusters=n_digits, n_init=10) 

n_clusters = n_digits            
X = reduced_data
labels_true = y_train
# Compute clustering with Means
#k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

# #############################################################################
# Plot result
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#F6352C', '#F67B2C', '#F6C52C', '#C3F62C', '#67F62C', '#2CF696', '#2CC4F6', '#060806', '#2C31F6','#DB2CF6']

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# KMeans
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())

plt.text(-4, .25, 'train time: %.2fs\ninertia: %f' % (
    t_batch, k_means.inertia_))

plt.show()