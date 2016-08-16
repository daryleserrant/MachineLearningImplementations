from numpy.linalg import lstsq

class NMF(object):
    def __init__(self, mat,k,num_iter):
        self.V = mat.todense().T
        self.k = k
        self.num_iter = num_iter
        self.W = np.random.randint(10, size=(self.V.shape[0],k))
        self.H = np.random.randint(10, size=(k, self.V.shape[1]))
        self.fitted_resids = np.inf
    def fit(self):
        counter = 0
        for i in xrange(self.num_iter):
            if i % 10 == 0:
                print i
            
            tmp_iter_1 = lstsq(self.W, self.V)
            self.H = np.clip(tmp_iter_1[0], 0., np.inf)
        
            tmp_iter_2 = lstsq(self.H.T, self.V.T)
            self.W = np.clip(tmp_iter_2[0].T, 0, np.inf)
            res = np.sum(tmp_iter_2[1], axis=0)
            self.fitted_resids = np.sum(tmp_iter_2[1], axis=0)
            
            self.fitted_resids = np.sum(tmp_iter_2[1], axis=1)
            if self.fitted_resids < 10.:
                break
                
        return self.W, self.H