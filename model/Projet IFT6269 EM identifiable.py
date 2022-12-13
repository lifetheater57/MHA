#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def ProjectNonNegative( W ):
    """
    projection onto the non-negative orthant
    """

    return W * ( W > 0 )


def ProjectMax1( W ):
    """
    project onto set of non-neg matricies with one nonzero entry per row
    """
    for i in range( W.shape[0] ):
        max_id = W[ i,: ].argmax()
        W[i,:][ np.where( W[ i,: ] < W[ i,max_id ] )[0] ] = 0
        W[i, max_id] = max(0, W[i, max_id])

    return W 

def normalizeColumns( W ):
    """
    standardize columns of W to be unit norm
    """
    for i in range( W.shape[1] ):
        W[:,i] /= ( np.linalg.norm( W[:,i] ) + .001 )

    return W


# In[3]:


from typing import Tuple, Union

import numpy as np

class GaussianGenerator(object):
    """ Generator for high-dimensional gaussian data
    
    Attributes
    ----------
    W : array([p, k])
        Factor loadings
    G : array([N, k, k])
        Covariance structures of latent variables
    v : array([N])
        Class-wise noise

    Yields
    ------
    X : array([N, p])
        Generated data (1 p-dimensional observation per class)

    """

    def __init__(self, N: int, p: int, k: int, seed: int, size: Union[int, Tuple[int]] = 1):
        """
        Parameters
        ----------
        N : int
            Nb of classes, aka number of observation classes sharing
            the same factor loadings
        p : int
            Dimensionality of the observations
        k : int
            Number of latent variables
        seed : int
            RNG seed
        size : int|tuple(int)
            The size of the samples generated for each class. The shape of the output is `N x size x p`.
        """
        # Fix seed for reproducibility
        np.random.seed(seed)
        
        epsilon = 1e-10

        self.N = N
        self.p = p
        self.k = k
        self.size = size

        # Create factor loading matrix W
        # Constraints on F: orthonormal and non-negative
        indexes = np.random.randint(0, k, size=p)
        self.W = np.random.uniform(0, 1, size=(p, k))
        #self.W = self.W * np.eye(k)[indexes]
        #self.W /= np.linalg.norm(self.W, axis=0)
        self.W = ProjectMax1(self.W)
        self.W = normalizeColumns(self.W)
        self.W = normalizeColumns(self.W)

        # Create latent variable covariance G_i
        # We use a triangular matrix to enforce a positive semi-definite covariance matrix
        self.G = [np.tril(X) @ np.tril(X).T for X in np.random.normal(size=(N, k, k))]

        self.v = np.random.normal(1, 1, size=N)**2
        # self.v = np.ones(N)

        # assert constraints on W and G
        # assert (np.linalg.norm(self.W, axis=0) - np.ones(k) < epsilon).all(), \
        #    "Columns of W are not unit vectors."
        # assert (np.abs((self.W.T @ self.W) - np.eye(k)) < epsilon).all(), \
        #    "W is not orthogonal."
        # assert np.array([G_i == G_i.T for G_i in self.G]).all(), \
        #    "Covariance matrices are not symmetrical."
        

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        Z = [np.random.multivariate_normal(np.zeros(G_i.shape[0]), G_i, size=self.size) for G_i in self.G]
        X = np.array([[np.random.multivariate_normal(self.W @ z_i, v_i * np.eye(self.p)) for z_i in Z_i] for Z_i, v_i in zip(Z, self.v)])
        return X


# In[4]:


gen = GaussianGenerator(10, 50, 5, 420, 500)
X = gen.next()
print(X.shape)


# In[13]:


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
        
    def fit(self, lr = 1e-3, reg = 1e-2, tol=1e-2, max_iter = 10):
        
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
        
            self.sigmas_z = np.array([self.G[i] - (self.W @ self.G[i]).T @  np.linalg.lstsq(self.sigmas[i], self.W @ self.G[i], rcond=None)[0] for i in range(self.N)])
        
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
            W_old = np.copy(self.W)
            iter_grad = 0
            
            while np.sum(np.abs(self.W - W_old)) < tol and iter_grad <= max_iter:
                
                for i in range(self.N):
                    for j in range(self.n):
                        
                        W_grad += 1/self.v[i]*(np.outer(self.X[i][j], self.E_z[i][j]) - self.W @ self.E_z_2[i][j])
                
                penalty = reg * (self.W @ self.W.T @ self.W - self.W)                                 
                ortho = self.W @ self.lagrange
            
                W_grad += penalty + ortho
                
                self.W -= lr * W_grad
                
                self.W = ProjectMax1(self.W)
                self.W = ProjectNonNegative(self.W) # to ensure non-negativity and orthonormality 
                self.W = normalizeColumns(self.W)
                
                self.lagrange += reg * (self.W.T @ self.W - np.eye(self.k))
                
                self.G = np.array([self.W.T @ self.K[i] @ self.W - self.v[i]*np.eye(self.k) for i in range(self.N)])
            
                self.sigmas = np.concatenate([(self.W @ self.G[i] @ self.W.T + self.v[i]*np.eye(self.p))[np.newaxis, ...] for i in range(self.N)])
                
                W_old = np.copy(self.W)
                
                iter_grad += 1
            

            
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


# In[14]:


k = 5
connect = ConnectivityEM(X, k)
print(connect.W.shape)
print(connect.G.shape)
print(connect.sigmas.shape)

# LL = connect.negative_log_likelihood()
# print(LL)

#for i in range(connect.N):
#    d, V = np.linalg.eig(connect.G[i])
#    v_i = connect.v[i]
#    A = V @ np.diag(d/(d + v_i)) @ V.T
#
#    D_tilde = V @ np.diag(-d/(d+v_i)**2) @ V.T
#
#    D_bar = V.T @ np.diag(-2*d**2/(d+v_i)**3) @ V.T
#    print("d ", d)
#    print("V ", V)
LLs = connect.fit()
# print(len(LLs))
#plt.xscale('log')
plt.plot(LLs)


# In[ ]:




