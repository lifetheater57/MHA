import numpy as np

class Gaussian_generator(object):
    """ Generator for high-dimensional gaussian data
    
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

    Attributes
    ----------
    W : array([p, k])
        Factor loadings
    Gs : array([N, k, k])
        Covariance structures of latent variables

    Yields
    ------
    X : array([N, p])
        Generated data (1 p-dimensional observation per class)

    """

    def __init__(self, N, p, k, seed):
        #TODO: create factor loading matrix W
        # constraints on F: orthonormal and non-negative

        #TODO: for i in classes
            #TODO: create latent variable covariance G_i

    # while True:
        #TODO: yield X