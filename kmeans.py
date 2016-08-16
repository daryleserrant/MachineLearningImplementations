import numpy as np
import sklearn.datasets as skdata
import random

def run_kmeans(data, k = 10, num_iter=10000):
    '''
    Perform K-Means clustering on the dataset
    
    Arguments:
       data - The dataset
       k - Number of clusters
       num_iter - Number of iterations
    
    Returns:
        The cluster assignment for each data point and the cluster centroids
    '''
    centroids = random.sample(data, k)
    assignment = np.zeros(len(data))
    iter_assign = np.ones(len(data))
    distances = np.zeros(k)
    niter = num_iter
    
    
    while((not np.array_equal(assignment, iter_assign)) and (niter > 0)):
        assignment = iter_assign.copy()
        
        for idx, point in enumerate(data):
            for loc,c in enumerate(centroids):
                distances[loc] = np.linalg.norm(point - c)
                
            iter_assign[idx] = np.argmin(distances)
        
        for ix in range(k):
            cluster = data[iter_assign == ix]
            centroids[ix] = np.mean(cluster)
        niter -= 1
    
    return (iter_assign, centroids)

if __name__ == "__main__":
    iris = skdata.load_iris()
    X = iris['data']
    labels, centroids = run_kmeans(X, 10)