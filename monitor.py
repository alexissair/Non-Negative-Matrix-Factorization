import matplotlib.pyplot as plt
from utils import *
from kul_mul import *
from fro_mul import *

class Monitor() :
    def __init__(self, algo, niter, metric, trained = False, opt = None) :
        self.algo = algo
        self.niter = niter
        self.opt = opt
        self.metric = metric
        self.trained = trained
        self.err = []
        return
    
    def train_algo(self):
        self.algo.train(self.niter)
        return
    
    def get_error(self):
        if not self.trained :
            self.train_algo()
        if self.metric == 'kl' :
            for i in range(len(self.algo.h_hist)) :
                V_est = self.algo.w_hist[i] @ self.algo.h_hist[i]
                self.err.append(kl(self.algo.V, V_est))
        elif self.metric == 'Frobenius' :
            for i in range(len(self.algo.h_hist)) :
                V_est = self.algo.w_hist[i] @ self.algo.h_hist[i]
                self.err.append(np.linalg.norm(self.algo.V -V_est, ord = 2))
        return self.err
    
    def plot_error(self):
        self.get_error()
        plt.figure()
        plt.plot(self.err)
        plt.title('Evolution of the error between V and WH, metric : {}'.format(self.metric))
        plt.xlabel('Number of iterations')
        plt.ylabel('Error')
        plt.show()
        return

if __name__ == "__main__":
    knmf = KullbachNMF(V = np.random.rand(10, 15), r = 7)
    monitor_kl = Monitor(algo = knmf, niter = 1000, metric = 'kl')
    monitor_kl.plot_error()

    
