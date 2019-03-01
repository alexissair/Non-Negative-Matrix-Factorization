import numpy as np
from fro_mul import *
from utils import *

class KullbachNMF(FrobeniusNMF) :
    def update_h(self) :
        self.h_hist.append(self.H)
        aux_h = self.W.T @ (self.V / (self.W @ self.H))
        self.H = self.H * aux_h 
        self.H = self.H / (self.W.T @ np.ones((self.nrows, self.ncols)))
        return

    def update_w(self) :
        self.w_hist.append(self.W)
        self.V_hist.append(self.W @ self.H)
        aux_w = (self.V / (self.W @ self.H)) @ self.H.T
        self.W = self.W * aux_w
        self.W = self.W / (np.ones((self.nrows,self.ncols)) @ self.H.T)
        return




