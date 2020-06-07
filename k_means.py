import numpy as np
import random
import sys
from matplotlib import pyplot as plt

# plot the current centroids and data pts with the centroid they belong to
def plot_k_means(title,assign,datum,means):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    for i in range(0,np.shape(assign)[0]):
        ax1.scatter(c=(0.005 * assign[i], 0.005 * assign[i], 0.005 * assign[i]), data = datum[i])
    ax1.scatter(data = means.T, c='r')

init = 1
k = 10
rand_r = random.randrange(1,init+1)
local_err = global_err = sys.float_info.max
optimal_centroids = optimal_assignment = np.zeros(1)
data = np.loadtxt('mnist_train.csv', delimiter=',')  # if loadtxt doesn't work, try genfromtxt

for r in range(1,init+1):
    local_err = it = assign_same = 0.0
    assignment = np.arange(np.shape(data)[0])
    # initialize centroid positions (means) 
    centroids = np.ones((k,np.shape(data)[1]))
    for i in range(0,np.shape(centroids)[0]):
        centroids[i] = data[random.randrange(np.shape(data)[0])]   
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
#plot_k_means('Optimal solution', optimal_assignment, data, optimal_centroids)
#plt.show()
