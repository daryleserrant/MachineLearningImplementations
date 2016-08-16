from __future__ import division
from collections import Counter, defaultdict
import numpy as np
import itertools
from math import log

class NaiveBayes(object):
    def __init__(self, alpha=1.):
        """
        INPUT:
        -alpha: float, laplace smoothing constant.
        ATTRIBUTES:
        - class_counts: the number of samples per class; keys=labels
        - class_feature_counts: the number of samples per feature per label;
                               keys=labels, values=Counter with key=feature
        - class_freq: the frequency of each class in the data
        - p: the number of features
        """
        self.class_counts = defaultdict(int)
        self.class_feature_counts = defaultdict(Counter)
        self.class_freq = None
        self.alpha = float(alpha)
        self.p = None

    def _compute_likelihoods(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels
        OUTPUT: None
        Compute the word count for each class and the frequency of each feature
        per class.  (Compute class_counts and class_feature_counts).
        '''
        num_rows = len(X)
        for row in xrange(num_rows):
            self.class_feature_counts[y[row]] += Counter(X[row])
            self.class_counts[y[row]] += len(X[row])

    def fit(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels
        OUTPUT: None
        '''
        # Compute class frequency P(y)
        self.class_freq = Counter(y)

        # Compute number of features
        self.p = len(set(itertools.chain(*X)))

        # Compute likelihoods
        self._compute_likelihoods(X, y)

    def posteriors(self, X):
        '''
        INPUT:
        - X: List of list of tokens.
        OUTPUT:
        List of dictionaries with key=label, value=log(P(y) * sum(P(x_i|y))).
        '''

        prob_dicts = []
        total_articles = sum(dict(self.class_freq).values())

        for row in X:
            prob = {}

            for lbl in self.class_freq.iterkeys():
                prob[lbl] = log(self.class_freq[lbl] / float(total_articles))
                for word, count in Counter(row).iteritems():
                    prob[lbl] += (log((self.class_feature_counts[lbl][word] + self.alpha) / float(self.class_counts[lbl]) + self.alpha*self.p)*count)
            prob_dicts.append(prob)

        return prob_dicts


    def predict(self, X):
        """
        INPUT:
        - X: A list of lists of tokens.
        OUTPUT:
        - predictions: a numpy array with predicted labels.
        """
        predictions = []
        for post in self.posteriors(X):
            pred = max(post.iterkeys(), key=(lambda label: post[label]))
            predictions.append(pred)
        return np.array(predictions)

    def score(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels
        OUTPUT:
        - accuracy: float between 0 and 1
        Calculate the accuracy, the percent predicted correctly.
        '''

        return np.mean(self.predict(X) == y)