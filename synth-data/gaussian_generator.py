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
        
        epsilon = 1e-10

        self.N = N
        self.p = p
        self.k = k

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
        assert (np.linalg.norm(self.W, axis=0) - np.ones(k) < epsilon).all(), \
            "Columns of W are not unit vectors."
        assert (np.abs((self.W.T @ self.W) - np.eye(k)) < epsilon).all(), \
            "W is not orthogonal."
        assert np.array([G_i == G_i.T for G_i in self.G]).all(), \
            "Covariance matrices are not symmetrical."
        

    def __iter__(self):
        return self

    def __next__(self):
        return self.next

    def next(self):
        Z = [np.random.multivariate_normal(0, G_i, size=self.p) for G_i in self.G]
        X = [np.random.multivariate_normal(self.W @ z_i, v_i * np.eye(self.k), size=self.p) for z_i, v_i in zip(Z, self.v)]
        return X

