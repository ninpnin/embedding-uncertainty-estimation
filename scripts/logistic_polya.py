import array
import numpy as np
from polyagamma import random_polyagamma

def v_omega(X, y, omega, sigma_prior):
    

def polyagamma_gibbs(beta_init, X, y, iterations=2, N=None, mu_prior=None, sigma_prior=None):
    N = len(X)
    beta = beta_init
    
    if mu_prior is None:
        mu_prior = np.zeros(X.shape[-1])
    if sigma_prior is None:
        sigma_prior = np.identity(X.shape[-1])
        
    
    print(sigma_prior)
    for r in range(iterations):
        # Get a 5 by 1 array of PG(1, 2) variates.
        xTbeta = X @ beta
        omega = random_polyagamma(np.ones(len(X)), xTbeta)
        print(omega)
        
        
        

X = np.random.rand(3,2)
y = np.array([1,0,1])
print(len(X))
beta_init = np.random.randn(2) * 0.2

polyagamma_gibbs(beta_init, X, y)