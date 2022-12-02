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
        No

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
        # Create factor loading matrix W
        # Constraints on F: orthonormal and non-negative
        np.random.seed(seed)
        indexes = np.random.randint(0, k, size=p)
        self.W = np.random.uniform(0, 1, size=(p, k))
        self.W = self.W * np.eye(k)[indexes]
        self.W /= np.linalg.norm(self.W, axis=0)

        # TODO: add assert constraints W is orthonormal, G is symmetric with negative and positive correlations

        # Create latent variable covariance G_i
        self.G = [np.tril(X) @ np.tril(X).T for X in np.random.normal(size=(N, k, k))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next

    def next(self):
        Z = np.random.multivariate_normal()


    def __init__(self, N, p, k, seed):
        #TODO: create factor loading matrix W
        # constraints on F: orthonormal and non-negative

        #TODO: for i in classes
            #TODO: create latent variable covariance G_i

    # while True:
        #TODO: yield X