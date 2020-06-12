import numpy as np
from numpy import loadtxt
import random
import sys
from matplotlib import pyplot as plt
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import metrics
from sklearn.decomposition import PCA

#### DATA PREP #### David's dimensionality reduction code
def get_reduced_data(source, data_size, data_max, num_clusters):
    # smaller number for faster convergence
    train_samples = data_size
    # Load data from https://www.openml.org/d/554
    #X, y = fetch_openml(source, version=1, return_X_y=True)
    data_set = np.loadtxt(source, delimiter=',')
    #X = data_set[:,1:]
    #y = data_set[:,:1]
    X = data_set[:,:]
    #print(X.shape, y.shape)
    
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    #y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    
    #X_train, X_test, y_train, y_test = train_test_split(
     #   X, y, train_size=train_samples, test_size=1000)
    X_train, X_test,  = train_test_split(
        X, train_size=train_samples, test_size=1000)
    
    X_train = X_train/data_max
    X_test = X_test/data_max
    
    data = X_train
    n_samples, n_features = data.shape
    #n_digits = len(np.unique(y_train))
    n_digits = num_clusters
    #labels = y_train
    
    sample_size = data_size
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
    # return the data, reduced to 2-dimensions
    return n_digits, reduced_data


# plot the current centroids and data pts with the centroid they belong to (only 
# works for 2-D data)
def plot_k_means(title,assign,data,means):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    cluster_colors = np.array(['#F6352C', '#F67B2C', '#F6C52C', '#C3F62C', '#67F62C', 
                               '#2CF696', '#2CC4F6', '#060806', '#2C31F6','#DB2CF6',
                               '#59CFAB', '#A5F109', '#09FCB4', '#23CE3A','#CBC493',
                               '#67ABCD', '#595959', '#F04117', '#1D0E4A','#C0AECD',
                               '#F1AFAC', '#BEC321', '#40CE2A', '#50CE67','#EA9A0A',
                               '#AE8090', '#98A6ED', '#B36895', '#192034','#4823DC'])
    for i in range(0,np.shape(assign)[0]):
        ax1.scatter(data[i][0], data[i][1], c=cluster_colors[assign[i]])
    x1, y1 = data.T
    x2, y2 = means.T
    ax1.scatter(x2,y2,c='r')
    #plt.show()

# K-means algorithm, takes as input data source (reduce it first),
# number of initializations, data_size to use, and max value of a data point
# Returns final centroids, assignments for each data point, and the reduced data
def k_means(data_source, inits, data_size, data_max, num_clusters, initial_centroids): 
    rand_r = random.randrange(1,inits+1)
    local_err = global_err = sys.float_info.max
    optimal_centroids = optimal_assignment = np.zeros(1)
    k, data = get_reduced_data(data_source, data_size, data_max, num_clusters)
    
    for r in range(1,inits+1):
        local_err = it = assign_same = 0.0
        assignment = np.arange(np.shape(data)[0])
        # initialize centroid positions (means) 
        centroids = np.ones((k,np.shape(data)[1]))
        temp_initial_centroids = np.copy(initial_centroids)
        temp_current_centroids = np.copy(centroids)
        while np.shape(temp_current_centroids)[0] > 0:
            rand_current_idx = random.randrange(np.shape(temp_current_centroids)[0])
            # initialize this runs centroids to previous passed in centroids (transfer learning)
            if np.shape(temp_initial_centroids)[0] > 0:
                rand_initial_idx = random.randrange(np.shape(temp_initial_centroids)[0])
                centroids[rand_current_idx] = temp_initial_centroids[rand_initial_idx]
                temp_initial_centroids = np.delete(temp_initial_centroids, rand_initial_idx, 0)
            else:
                centroids[rand_current_idx] = data[random.randrange(np.shape(data)[0])]   
            temp_current_centroids = np.delete(temp_current_centroids, rand_current_idx, 0)
        # run algorithm until every points cluster assignment doesn't change
        while assign_same < np.shape(data)[0]:
            assign_same = 0
            # assign each data point to nearest cluster
            for i in range(0,np.shape(data)[0]):
                prev_assignment = assignment[i] 
                assignment[i] = np.argmin(np.sum(pow(data[i]-centroids, 2), axis=1))
                # for assignments that didn't change, increment counter
                if assignment[i] == prev_assignment:
                    assign_same = assign_same + 1
            # update centroid values
            for i in range(0,np.shape(centroids)[0]):
                if np.count_nonzero(assignment == i) == 0:
                    centroids[i] = np.sum(data[j] for j in range(0, np.shape(data)[0]) if assignment[j] == i)
                else:
                    centroids[i] = np.sum(data[j] for j in range(0, np.shape(data)[0]) if assignment[j] == i) / np.count_nonzero(assignment == i)
                # get sum of square error for this r-initialization run
                if assign_same >= np.shape(data)[0]:
                    local_err = local_err + np.sum(np.sum(pow(data[j] - centroids[i],2) for j in range(0, np.shape(data)[0]) if assignment[j] == i))
            #if r == rand_r:
                #plot_k_means('Init num ' + str(r) + ' iteration ' + str(it), assignment, data, centroids)
            it = it + 1 
            
        # update optimal set of centroids and assignments after each r value run
        if local_err < global_err:
            global_err = local_err
            optimal_centroids = np.copy(centroids)
            optimal_assignment = np.copy(assignment)
    # give global error and plot data with optimal centroids (max of 10 clusters)
    print("After", r, "initializations with k =", k, ", k-means has a sum squared error of", global_err)
    return optimal_centroids, optimal_assignment, data
    #plot_k_means('Optimal solution', optimal_assignment, data, optimal_centroids)
    #plt.show()

# ************MAIN**************

# compute and plot k-means for mnist
centroids = np.array([])
#start = time.time()
centroids, assignment, data = k_means('mnist_test-set.csv', 10, 30000, 255, 10, centroids)
#end = time.time()
#print("k-means took", end-start, "seconds")
#plot_k_means('K-means', assignment, data, centroids) 

# compute and plot k-means for letter
#centroids = np.array([])
start = time.time()
centroids, assignment, data = k_means('letter-recognition-set.csv', 10, 30000, 255, 26, centroids)
end=time.time()
print("k-means took", end-start, "seconds")
plot_k_means('K-means', assignment, data, centroids) 

plt.show()
