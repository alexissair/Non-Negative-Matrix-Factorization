import matplotlib.pyplot as plt
from utils import *
from kul_mul import *
from fro_mul import *

class Monitor() :
    def __init__(self, algos, niter, metric, trained = False, opt = None) :
        self.algos = algos
        self.niter = niter
        self.opt = opt
        self.metric = metric
        self.trained = trained
        self.err = np.zeros((len(self.algos), self.niter))
        return
    
    def train_algo(self):
        for algo in self.algos :
            algo.train(self.niter, verbose = False)
        return
    
    def get_error(self):
        if not self.trained :
            self.train_algo()
        if self.metric == 'kl' :
            for j in range(len(self.algos)) :
                for i in range(len(self.algos[j].h_hist)) :
                    self.err[j,i] = kl(self.algos[j].V, self.algos[j].V_hist[i])
        elif self.metric == 'Frobenius' :
            for j in range(len(self.algos)) :
                for i in range(len(self.algos[j].h_hist)) :
                    self.err[j,i] = np.linalg.norm(self.algos[j].V_hist[i] - self.algos[j].V, ord = 2)
        return self.err
    
    def plot_error(self, labels):
        assert len(labels) == len(self.algos)
        err = self.get_error()
        plt.figure()
        for i in range(len(self.algos)) :
            plt.plot(err[i,:], label = labels[i])
            plt.title('Evolution of the error between V and WH, metric : {}'.format(self.metric))
            plt.xlabel('Number of iterations')
            plt.ylabel('Error')
        plt.legend()
        plt.show()
        return

if __name__ == "__main__":
    V = np.random.rand(5000, 300)
    fronmf = KullbachNMF(V = V, r = 20)
    fronmf2 = KullbachNMF(V = V, r = 20)
    monitor_kl = Monitor(algos = [fronmf, fronmf2], niter = 100, metric = 'kl')
    monitor_kl.plot_error(labels = ['1', '2'])

    
