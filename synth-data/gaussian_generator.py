import numpy as np

class Gaussian_generator(self, N, N_size, p, k, seed):
    """ Gaussian generator for multivariate data
    
    Parameters
    ----------
    N : int
        Nb of classes, aka number of observation classes sharing
        the same factor loadings
    N_size: int
        Number of observations per class
    p : int
        Dimensionality of the observations
    k : int
        Number of latent variables
    seed : int
        RNG seed

    Returns
    -------
    X : array([N, size, p])
        Generated data
    F : array([p, k])
        Factor loadings
    Gs : array([N, k, k])
        Covariance structures of latent variables

    """
    #TODO: create factor loading matrix F
    
    #TODO: for i in classes
        #TODO: create latent variable covariance G_i
        #TODO: generate N_size values in latent space : [N_size, k]
        #TODO: broadcast these values in observation space : [N_size, p]

