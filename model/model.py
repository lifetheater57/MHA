def relu(x):
    
    return np.maximum(0,x)

class Connectivity:
    
     def __init__(self, data, k):
            
        self.n_ex = data.shape[0]
        self.n_dims = data.shape[1]
        
        self.data = data
        
        self.k = k 
        
        self.v = np.ones(self.n_ex)
            
        self.W = np.zeros((self.n_dims, k))
            
        self.G = np.concatenate([np.eye(k)[np.newaxis, ...]\
                                 for i in range(self.n_ex)], axis=0)
        
        
        self.sigmas = np.concatenate([self.W @ self.G[i] @ self.W\
                      + self.v[i]*np.eye(self.n_dims) for i in range(self.n_ex)])
        
        self.lagrange = np.eye(k)
    
    def log_likelihood(self):
        
        log_likelihood = 0
        
        for i in range(self.n_ex):
            
            cst = self.n_dims * np.log(2*np.pi)
            
            det_sigma = np.log(np.linalg.det(self.sigmas[i]))
            
            mahalanobis = self.data[i] @ np.linalg.lstsq(self.sigmas[i],\
                          self.data[i], rcond=None)[0]
            
            log_likelihood += -1/2*(cst + det_sigma + mahalanobis)
        
        
    def fit(self, lr = 1e-3, reg = 1e-2):
        
        old_likelihood = -1e2
        new_likelihood = 0
        
        W_grad = np.zeros((self.n_dims, self.k))
        
        while np.abs(new_likelihood - old_likelihood) > 1e-2:
            
            for i in range(self.n_ex):
                
                v_i = self.v[i]
                
                d, V = np.linalg.eig(self.G[i])
                
                A = V @ np.diag(d/(d+v_i)) @ V.T
                
                D_tilde = V @ diag(-d/(d+v_i)**2) @ V.T
                
                D_bar = V.T @ diag(-2*d**2/(d+v_i)**3) @ V.T
                
                H_1 = 2*A - v_i*D_tilde - A @ A + v_i * D_bar
                
                H_2 = -v_i**-2 * np.trace(A - v_i*D_tilde)
                
                K = np.outer(self.data[i], self.data[i])
                
                # G_i update
                                              
                self.G[i] = W.T @ K @ W - v_i * np.eye(self.n_dims) 
                
                # v_i update gradient descent
                
                v_term_1 = v_i**-3 * np.trace(v_i* np.eye(self.n_dims) - K
                v_term_2 = v_i**-3 * np.trace(W.T @ K @ W @ H_1)
                
                self.v[i] -= lr * (v_term_1 + v_term_2 - H_2)
                  
                # add ith term to W gradient
                
                W_grad += v_i**-2 * K * self.W(1/2 * A @ A - A)
                                              
            # update W                                 
            penalty = reg * (W @ W.T @ W - W)                                 
            ortho = W * self.lagrange
                                              
            W = relu(W - lr(W_grad + penalty + ortho))
                                              
            lagrange += reg * (W.T @ W - np.eye(self.k))
                                              
            # update sigmas using all other updates
            self.sigmas =  np.concatenate([self.W @ self.G[i] @ self.W\
                           + self.v[i]*np.eye(self.n_dims) for i in range(self.n_ex)])
                                              
            old_likelihood = new_likelihood
            new_likelihood = self.log_likelihood()
                                              
                                              
    def print_params(self):
        
        print('v:','\n', self.v, '\n', 'G:', '\n', self.G, '\n',\
             'W:', '\n', self.W, '\n', 'sigmas:', '\n', self.sigmas)