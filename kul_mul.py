import numpy as np
from fro_mul import *
from utils import *

class KullbachNMF(FrobeniusNMF) :
    def update_h(self) :
        self.h_hist.append(self.H)
        wh = self.W @ self.H
        for a in range(self.H.shape[0]):
            for mu in range(self.H.shape[1]) :
                s = 0
                for i in range(self.nrows) :
                    s += self.W[i, a] * self.V[i, mu] / wh[i, mu]
                self.H[a, mu] = self.H[a, mu] * s
                self.H[a, mu] /= np.sum(self.W[:, a])
        return

    def update_w(self) :
        self.w_hist.append(self.W)
        wh = self.W @ self.H
        self.V_hist.append(wh)
        for i in range(self.W.shape[0]):
            for a in range(self.W.shape[1]) :
                s = 0
                for mu in range(self.ncols) :
                    s += self.H[a, mu] * self.V[i, mu] / wh[i, mu]
                self.W[i, a] = self.W[i, a] * s
                self.W[i, a] /= np.sum(self.H[a, :])
        return




