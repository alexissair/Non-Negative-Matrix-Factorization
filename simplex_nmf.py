import numpy as np
from scipy.optimize import linprog

class SimplexNMF():
    def __init__(self, f=20, r=3, n=50, X=None):
        self.r = r
        if X is None:
            self.f = f
            self.n = n
            F = np.random.random((f, r))
            F = SimplexNMF.norm_rows_to_one(F)
            W = np.random.random((r, n))
            W = SimplexNMF.norm_rows_to_one(W)
            self.X = F @ W
        else:
            self.f, self.n = X.shape
            self.X = X

    @staticmethod
    def norm_rows_to_one(matrix):
        return matrix / np.expand_dims(np.sum(matrix, axis=1), -1)

    def build_a_eq(self):
        A = np.zeros((self.f*self.n + 1, self.f**2))
        for i in range(self.f):
            # Handle CX = X
            A[i*self.n:(i+1)*self.n, i*self.f:(i+1)*self.f] = self.X.T
            # Handle Tr(C) = r
            A[-1, i*(self.f+1)] = 1
        self.A_eq = A
    
    def build_b_eq(self):
        b = np.zeros((self.f*self.n+1,))
        # Handle CX = X
        b[:-1] = self.X.flatten()
        # Handle Tr(C) = r
        b[-1] = self.r
        self.b_eq = b

    def build_a_ub(self):
        A = np.zeros((2*self.f**2, self.f**2))
        index_A = 0

        # Handle diag(C) <= 1
        for i in range(self.f):
            A[i, i*(self.f+1)] = 1
            index_A += 1
            
        # Handle C_ij - C_jj <= 0  
        for j in range(self.f):
            for i in range(self.f):
                if i == j:
                    continue
                C = np.zeros((self.f, self.f))
                C[j, j] = -1
                C[i, j] = 1
                A[index_A, :] = C.flatten()
                index_A += 1

        # Handle -C_ij <= 0
        for i in range(self.f):
            for j in range(self.f):
                A[index_A, i*self.f + j] = -1
                index_A += 1
        
        self.A_ub = A

    def build_b_ub(self):
        # Handle C_ij - C_jj <= 0 and -C_ij <= 0
        b = np.zeros(2*self.f**2)
        # Handle diag(C) <= 1
        b[:self.f] += 1
        self.b_ub = b
    
    def build_t(self):
        self.t = (np.random.random((self.f, self.f)) * np.eye(self.f)).flatten()

    def run(self):
        self.build_a_eq()
        self.build_b_eq()
        self.build_a_ub()
        self.build_b_ub()
        self.build_t()
        self.result = linprog(self.t, method='simplex', A_eq=self.A_eq, b_eq=self.b_eq, A_ub=self.A_ub, b_ub=self.b_ub)
        self.c = self.result.x.reshape(self.f, self.f)
        self.build_nmf()
        self.check_result()

    def build_nmf(self):
        dia = np.diag(self.c)
        index = np.argpartition(dia, -self.r)[-self.r:]
        self.W = self.X[index, :]
        self.F = self.c[:, index]

    def check_result(self):
        print('||CX - X|| = {}'.format(np.sum(np.square(self.c @ self.X - self.X))))
        print('diag(C) = {}'.format(np.diag(self.c)))
        print('||FW - X|| = {}'.format(np.sum(np.square(self.F @ self.W - self.X))))
