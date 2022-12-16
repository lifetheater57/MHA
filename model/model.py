from model.maths import *


from scipy.stats import multivariate_normal

class Connectivity:
    
    def __init__(self, X, k):
        """
        Parameters
        ----------
        X : NumPy array of size N x m x p where N is the number of subjects, 
            m is the number of observations and p the dim of observartion space
        k : int
            Nb of latent variables
        """
            
        self.N, self.n, self.p = X.shape
        
        self.X = X
        
        self.k = k 
        
        self.v = np.ones(self.N) 
                
        self.G = np.concatenate([np.eye(k)[np.newaxis, ...]\
                                 for i in range(self.N)], axis=0)
        
        self.A = np.array([self.G[i] @ np.linalg.pinv(self.G[i] + np.eye(self.k)) for i in range(self.N)])
        
        self.K = np.array([self.X[i].T @ self.X[i] for i in range(self.N)])
        
        self.initialize_W()
         
        self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T\
                                       + np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
        
        self.lagrange = np.zeros((self.k,self.k))
        
    def initialize_W(self):
        X_cov = np.zeros((self.p, self.p))
        
        for i in range(self.N):
            
            X_cov += 1/self.N * self.K[i]
        
        # initialize W
        evd_X_cov = np.linalg.eig(X_cov)
        
        self.W = evd_X_cov[1][:, evd_X_cov[0].argsort()[::-1][:self.k]]
        
        # since evectors are sign invariant
        for i in range(self.W.shape[1]):
            if np.sum(self.W[:,i]) < 0:
                self.W[:,i] *= -1
                
        self.W = relu(self.W)

    def negative_log_likelihood(self, X):
        log_likelihood = 0
        
        for i in range(self.N):
            cst = self.p * np.log(2*np.pi)
            
            det_sigma = np.log(np.linalg.det(self.sigmas[i]))
            
            S = np.linalg.lstsq(self.sigmas[i], X[i].T, rcond=None)[0]
            
            mahalanobis = np.einsum('ij, ji -> i', X[i], S)
            
            log_likelihood += 0.5*(cst + det_sigma + mahalanobis)
        
        return log_likelihood.sum() / np.prod(X.shape[:2])
    

    def update_v(self, lr):
        for i in range(self.N):
            d, V = np.linalg.eig(self.G[i])
            D_tilde = V @ np.diag(-d/(d+self.v[i])**2) @ V.T
            D_bar = V.T @ np.diag(-2*d**2/(d+self.v[i])**3) @ V.T
            H_1 = 2*self.A[i] - self.v[i]*D_tilde - self.A[i] @ self.A[i] + self.v[i] * D_bar
            H_2 = -self.v[i]**-2 * np.trace(self.A[i] - self.v[i]*D_tilde)  
            v_term_1 = self.v[i]**-3 * np.trace(self.v[i]* np.eye(self.p) - self.K[i])
            v_term_2 = self.v[i]**-3 * np.trace(self.W.T @ self.K[i] @ self.W @ H_1)
            
            self.v[i] -= lr * (v_term_1 + v_term_2 + H_2)
        
    def update_A(self, W):
        
        #W = project_positive(self.W).copy()
        
        for i in range(self.N):
            A_new = np.eye(self.k) - self.v[i]*np.linalg.pinv(W.T @ self.K[i] @ W)
        
            # check for negative eigenvalues
            if np.min(np.linalg.eig(A_new)[0]) <= 0:
                A_new += np.eye(self.k)* np.abs(np.min(np.linalg.eig(A_new)[0]) + 0.001)
            
            self.A[i] = A_new
    
    def armijo_Update(self, W_grad, A_tilde, alpha = 0.5, c = 0.001, max_iter = 100):
        
        stopBackTracking = False
        W_grad = normalize_cols(W_grad)
        iter = 0
        
        while stopBackTracking==False:
            W_new = self.W - alpha * W_grad
            
            W_new = relu(W_new)
            W_new = normalize_cols(W_new)
            
            old_obj = 0
            new_obj  = 0
            
            for i in range(self.N):
                old_obj += np.diag(self.W.T @ self.K[i] @ self.W @ A_tilde[i]).sum()
                new_obj += np.diag(W_new.T @ self.K[i] @ W_new @ A_tilde[i]).sum()
                
            if new_obj <= old_obj + c*alpha*(np.diag(np.diag(W_grad.T @ (W_new - self.W))).sum() + 0.001):
                stopBackTracking = True
                
            else:
                alpha /= 2
                iter += 1
                
                if iter > max_iter:
                    stopBackTracking = True
                
        return W_new
    
    def fit(self, lr = 0.001, reg = 1, c = 0.01, alpha = 0.5, tol=1e-2, max_iter = 1000, use_armijo=True, estimate_v=False):
        
        # Inits
        new_likelihood = float('-inf')
        old_likelihood = 0
        log_likelihoods = []
        W_old = np.copy(self.W)
        
        
        self.update_A(self.W) # Update A once before the loop
        for iter in range(max_iter):
            A_tilde = np.array([0.5*A_i @ A_i - A_i for A_i in self.A])
            
            # Compute W_grad
            W_grad = np.zeros(self.W.shape)
            
            for i in range(self.N):
                W_grad += self.v[i] * self.K[i] @ self.W @ A_tilde[i] / self.N
            
            penalty = reg * (self.W @ self.W.T @ self.W - self.W)                                 
            ortho = self.W @ self.lagrange
            W_grad += penalty + ortho
            
            # Compute gradient update
            if use_armijo:
                self.W = self.armijo_Update(W_grad, A_tilde, alpha, c, max_iter)
            else:
                self.W -= lr * W_grad
            
            # Make sure W respects identifiability constraints
            self.W = project_positive(self.W)
            self.W = relu(self.W) # to ensure non-negativity and orthonormality 
            self.W = normalize_cols(self.W)
            
            # Update A

            self.update_A(normalize_cols(project_positive(self.W.copy())))

            # Update v if necessary
            if estimate_v:
                self.update_v(lr)
            
            self.lagrange += reg * (self.W.T @ self.W - np.eye(self.k))
            
            # update G and sigmas
            self.G = np.array([self.W.T @ self.K[i] @ self.W - np.eye(self.k) for i in range(self.N)])
            
            self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T\
                                       + np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
            
            if np.sum(np.abs(self.W - W_old)) < tol:
                break
            
            else:
                W_old = np.copy(self.W)

            # Log information    
            old_likelihood = new_likelihood
            new_likelihood = self.negative_log_likelihood(self.X)
            log_likelihoods.append(new_likelihood)

        # compute final parameters
        self.W = normalize_cols(project_positive(self.W))
        self.G = np.array([self.W.T @ self.K[i] @ self.W - np.eye(self.k) for i in range(self.N)])

        return log_likelihoods
                                              
    def print_params(self):
        # print("v:\n", self.v)
        print("G:\n", self.G)
        print("W\n", self.W)
        print("sigmas\n", self.sigmas)



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
        
        self.G = np.array([self.W.T @ self.K[i] @ self.W - np.eye(self.k) for i in range(self.N)])
        self.K = np.array([self.X[i].T @ self.X[i] for i in range(self.N)])
        
        self.initialize_W()
            
        self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T + self.v[i]*np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
        
        self.lagrange = np.zeros((self.k,self.k))
        
    def initialize_W(self):
        
        X_cov = np.zeros((self.p, self.p))
        
        for i in range(self.N):
            
            X_cov += 1/self.N * self.K[i]
        
        # initialize W
        evd_X_cov = np.linalg.eig(X_cov)
        
        self.W = evd_X_cov[1][:, evd_X_cov[0].argsort()[::-1][:self.k]]
        
        # since evectors are sign invariant
        for i in range(self.W.shape[1]):
            if np.sum(self.W[:,i]) < 0:
                
                self.W[:,i] *= -1
                
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
