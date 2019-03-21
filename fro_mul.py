import numpy as np


class FrobeniusNMF() :
    def __init__(self, V, r, W_init = None, H_init = None) :
        assert V.all() >= 0
        self.V = V
        self.r = r
        self.nrows, self.ncols = V.shape[0], V.shape[1]
        if W_init is not None:
            self.W = W_init
        else :
            self.W = np.random.rand(self.nrows, self.r)
        if H_init is not None:
            self.H = H_init
        else :
            self.H = np.random.rand(self.r, self.ncols)
        self.w_hist = []
        self.h_hist = []
        self.V_hist = []
        return

    def update_h(self) :
        self.h_hist.append(self.H)
        wv = self.W.T @ self.V
        wwh = self.W.T @ self.W @ self.H
        for a in range(self.H.shape[0]):
            for mu in range(self.H.shape[1]) :
                self.H[a, mu] = self.H[a, mu] * wv[a, mu] / wwh[a, mu]
        return

    def update_w(self) :
        self.w_hist.append(self.W)
        self.V_hist.append(self.W @ self.H)
        vh = self.V @ self.H.T
        whh = self.W @ self.H @ self.H.T
        for i in range(self.W.shape[0]):
            for a in range(self.W.shape[1]) :
                self.W[i, a] = self.W[i, a] * vh[i, a] / whh[i, a]
        return

    def train(self, niter, verbose) :
        for i in range(niter):
            if verbose :
                if i%10 == 0 :
                    print('Iter n° {}'.format(i))
            self.update_h()
            self.update_w()
        return

    
    




    