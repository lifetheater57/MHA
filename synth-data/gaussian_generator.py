import numpy as np

class Gaussian_generator(object):
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

    def __init__(self, N, p, k, seed):
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
        """
        # Fix seed for reproducibility
        np.random.seed(seed)
        
        # Create factor loading matrix W
        # Constraints on F: orthonormal and non-negative
        indexes = np.random.randint(0, k, size=p)
        self.W = np.random.uniform(0, 1, size=(p, k))
        self.W = self.W * np.eye(k)[indexes]
        self.W /= np.linalg.norm(self.W, axis=0)

        # Create latent variable covariance G_i
        # We use a triangular matrix to enforce a positive semi-definite covariance matrix
        self.G = [np.tril(X) @ np.tril(X).T for X in np.random.normal(size=(N, k, k))]
        
        self.v = np.random.normal(size=N)

        # assert constraints on W and G
        assert all(np.linalg.norm(self.W, axis=0) == np.ones(k)), \
            "Columns of W are not unit vectors."
        assert all(self.W.T @ self.W == np.eye(k)), \
            "W is not orthogonal."
        assert all([G_i == G_i.T for G_i in G]), \
            "Covariance matrices are not symmetrical."
        

    def __iter__(self):
        return self

    def __next__(self):
        return self.next

    def next(self):
        pass
        # not finish
        #self.Z = np.random.multivariate_normal(0, self.G)

