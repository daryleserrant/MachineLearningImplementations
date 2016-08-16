from DecisionTree import DecisionTree
from collections import Counter
import numpy as np

class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        forest = []
        for b in xrange(num_trees):
            boot_idx = np.random.choice(len(X), len(X))
            boot_X = X[boot_idx]
            boot_y = y[boot_idx]
            tree = DecisionTree(num_features)
            tree.fit(boot_X, boot_y)
            forest.append(tree)
        return forest

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
        predictions = []
        tree_preds = np.empty((0,X.shape[1]))
        for tree in self.forest:
            tree_preds = np.vstack((tree_preds,tree.predict(X)))
        
        for obs in tree_preds.T:
            votes = Counter(obs)
            predictions.append(votes.most_common(1)[0][0])
        
        return np.array(predictions)

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        predictions = self.predict(X)
        yCounter = Counter(y)
        
        accuracy = {}
        for label,counts in yCounter.iteritems():
            numpos = sum([1 if (pred == cls) and (cls == label) else 0 for pred,cls in zip(predictions,y)])
            accuracy[label] = numpos / float(counts)
            
        return accuracy
