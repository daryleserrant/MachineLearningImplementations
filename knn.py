import numpy as np
import sklearn.datasets as skd

def euclidean_distance(p, q):
    return np.sqrt(((q - p)**2).sum()) 

def cosine_distance(p, q):
    return 1 - (np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q)))

class KNearestNeighbors(object):
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
        self.featureMatrix = np.array([])
        self.labels = np.array([])
    
    def fit(self, X, y):
        self.featureMatrix = X
        self.labels = y
    
    def predict(self, X):
        predictions = []
        for point in X:
            distDict = {idx:self.distance(point, feature) for idx,feature in enumerate(self.featureMatrix)}
            knn = self.labels[sorted(distDict, key=distDict.__getitem__)[:self.k]]
            npos = knn.sum()
            nneg = len(knn[knn == 0])
            if npos > nneg:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        numpos = float(y.sum())
        numneg = float(len(y) - numpos)
        tp = sum([1 if (lab == pred) and (lab == 1) else 0 for pred,lab in zip(predictions, y)])
        tn = sum([1 if (lab != pred) and (lab == 0) else 0 for pred,lab in zip(predictions, y)])
        return (tp + tn)/(numpos + numneg)

if __name__ == "__main__":
    X, y = skd.make_classification(n_features=4, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, class_sep=5, random_state=5)
    knn = KNearestNeighbors(k=3, distance=euclidean_distance)
    knn.fit(X, y)
    y_predict = knn.predict(X)
    print y_predict