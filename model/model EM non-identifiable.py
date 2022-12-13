class ConnectivityEM:
    
    def __init__(self, X, k):
        """
        Parameters
        ----------
        X : NumPy array of size N x m x k where N is the number of subjects, 
            m is the number of observations and p the dim of observartion space
        k : int
            Nb of latent variables
        """
            
        self.N, self.n, self.p = X.shape
        
        self.X = X
        
        self.k = k 
        
        self.v = np.ones(self.N) # change to init variance of samples?
        
        self.K = np.array([self.X[i].T @ self.X[i] for i in range(self.N)])
        
        self.initialize_W()
            
        self.G = np.array([self.W.T @ self.K[i] @ self.W - np.eye(self.k) for i in range(self.N)])
         
        self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T + self.v[i]*np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
        
        self.lagrange = np.zeros((self.k,self.k))
        
    def initialize_W(self):
        
        X_cov = np.zeros((self.p, self.p))
        
        for i in range(self.N):
            
            X_cov += 1/self.N * self.K[i]
        
        # initialize W
        evd_X_cov = np.linalg.eig(X_cov)
        
        self.W = evd_X_cov[1][:, evd_X_cov[0].argsort()[::-1][:k]]
        
        # since evectors are sign invariant
        for i in range(self.W.shape[1]):
            if np.sum(self.W[:,i]) < 0:
                
                self.W[:,i] *= -1
                
        #self.W = ProjectNonNegative(self.W)
    
    def negative_log_likelihood(self):

        log_likelihood = 0
        
        for i in range(self.N):
            
            cst = self.p * np.log(2*np.pi)
            
            det_sigma = np.log(np.linalg.det(self.sigmas[i]))
            
            S = np.linalg.lstsq(self.sigmas[i], self.X[i].T, rcond=None)[0]
            
            mahalanobis = np.einsum('ij, ji -> i', self.X[i], S)
            
            log_likelihood += 0.5*(cst + det_sigma + mahalanobis)
        
        return log_likelihood.sum()/(self.n * self.N)
        
    def fit(self, lr = 1e-3, reg = 1e-2, tol=1e-2):
        
        old_likelihood = np.inf
        new_likelihood = 0
        
        
        log_likelihoods = []
        iter = 0
        
        while np.abs(new_likelihood - old_likelihood) > tol:
            print(f"iteration {iter}")
            print("LL =",new_likelihood)
            iter += 1
            W_grad = np.zeros(self.W.shape)
            
            # E-step
            self.mu_z = np.array([[(self.W @ self.G[i]).T @ np.linalg.lstsq(self.sigmas[i], self.X[i][j], rcond=None)[0] for j in range(self.n)] for i in range(self.N)])
        
            self.sigmas_z = np.array([self.G[i] - (self.W @ self.G[i]).T @ np.linalg.lstsq(self.sigmas[i], self.W @ self.G[i], rcond=None)[0] for i in range(self.N)])
        
            self.E_z = self.mu_z
            
            self.E_z_2 = np.array([[self.sigmas_z[i] + np.outer(self.E_z[i][j], self.E_z[i][j]) for j in range(self.n)] for i in range(self.N)])
            
            # M-step
            for i in range(self.N):
                for j in range(self.n):
                    self.G[i] += self.E_z_2[i][j]
                    
                    self.v[i] += (0.5*np.sum(self.X[i][j]**2) - self.E_z[i][j] @ self.W.T @ self.X[i][j]) + 0.5*np.trace(self.E_z_2[i][j] @ self.W.T @ self.W)
                                   
                self.G[i] /= self.n
                                   
                self.v[i] *= 2/(self.p * self.n)
                
            
            # W update
            term_1_W = 0
            term_2_W = 0
                                   
            for i in range(self.N):
                for j in range(self.n):
                    
                    term_1_W += np.outer(self.X[i][j], self.E_z[i][j])
                    term_2_W += self.E_z_2[i][j]
                                   
            self.W = term_1_W @ np.linalg.pinv(term_2_W)
            
            self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T + self.v[i]*np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
        
            old_likelihood = new_likelihood
            new_likelihood = self.negative_log_likelihood()
            log_likelihoods.append(new_likelihood)

        return log_likelihoods
                                              
    def print_params(self):
        print("v:\n", self.v)
        print("G:\n", self.G)
        print("W\n", self.W)
        print("sigmas\n", self.sigmas)
