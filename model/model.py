def relu(x):
    return np.maximum(0,x)

class Connectivity:
    
    def __init__(self, X, k):
        """
        Parameters
        ----------
        X : NumPy array of size N x m x k where N is the number of subjects, 
            m is the number of observations and p the dim of observartion space
        k : int
            Nb of latent variables
        """
            
        self.N, self.m, self.p = X.shape
        
        self.X = X 
        
        self.k = k 
        
        self.v = np.ones(self.N) # change to init variance of samples?
            
        self.W = np.zeros((self.p, self.k))
            
        self.G = np.concatenate([np.eye(k)[np.newaxis, ...]\
                                 for i in range(self.N)], axis=0) 

        self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T\
                                       + self.v[i]*np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
        
        self.lagrange = np.eye(k)

    def log_likelihood(self):
        
        log_likelihood = 0
        
        for i in range(self.N):
            
            cst = self.p * np.log(2*np.pi)
            
            det_sigma = np.log(np.linalg.det(self.sigmas[i]))
            
            S = np.linalg.lstsq(self.sigmas[i], self.X[i].T, rcond=None)[0]
            
            mahalanobis = np.einsum('ij, ji -> i', self.X[i], S)
            
            log_likelihood += -0.5*(cst + det_sigma + mahalanobis)
        
        return log_likelihood.sum()
        
    def fit(self, lr = 1e-3, reg = 1e-2, tol=1e-6):
        
        old_likelihood = np.inf
        new_likelihood = 0
        
        W_grad = np.zeros((self.p, self.k))
        log_likelihoods = []
        while np.abs(new_likelihood - old_likelihood) > tol:
            for i in range(self.N):
                v_i = self.v[i]
                
                d, V = np.linalg.eig(self.G[i])
                
                A = V @ np.diag(d/(d + v_i)) @ V.T
                
                D_tilde = V @ np.diag(-d/(d+v_i)**2) @ V.T
                
                D_bar = V.T @ np.diag(-2*d**2/(d+v_i)**3) @ V.T
                
                H_1 = 2*A - v_i*D_tilde - A @ A + v_i * D_bar
                
                H_2 = -v_i**-2 * np.trace(A - v_i*D_tilde)
                
                K = self.X[i].T @ self.X[i]
                    
                # G_i update
                self.G[i] = self.W.T @ K @ self.W - v_i * np.eye(self.k) 
                
                # v_i update gradient descent
                v_term_1 = v_i**-3 * np.trace(v_i* np.eye(self.p) - K)
                v_term_2 = v_i**-3 * np.trace(self.W.T @ K @ self.W @ H_1)
                
                self.v[i] -= lr * (v_term_1 + v_term_2 - H_2)
                # add ith term to W gradient
                
                W_grad += v_i**-2 * K @ self.W @ (1/2 * A @ A - A)
                                              
            # update W                                 
            penalty = reg * (self.W @ self.W.T @ self.W - self.W)                                 
            ortho = self.W @ self.lagrange
                                              
            self.W = relu(self.W - lr*(W_grad + penalty + ortho)) # projection onto the non-negative orthant
                                              
            self.lagrange += reg * (self.W.T @ self.W - np.eye(self.k))
                                              
            # update sigmas using all other updates
            self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T\
                                       + self.v[i]*np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
                                              
            
            old_likelihood = new_likelihood
            new_likelihood = self.log_likelihood()
            log_likelihoods.append(new_likelihood)

        return log_likelihoods
                                              
    def print_params(self):
        
        print('v:','\n', self.v, '\n', 'G:', '\n', self.G, '\n',\
             'W:', '\n', self.W, '\n', 'sigmas:', '\n', self.sigmas)

