import numpy as np
from polyagamma import random_polyagamma
import progressbar

DIM = 2
DATALEN = 150
np.random.seed(124)

def sigmoid(x):
    y_inv = 1 + np.exp(-x)
    return 1 / y_inv

beta = 3 * np.random.randn(DIM) / np.sqrt(DIM)
print(beta)
X = np.random.randn(DATALEN,DIM)
etas = beta @ X.T
probs = sigmoid(etas)
print("eta", etas[:3])
print("p  ", probs[:3])

y = np.random.binomial(1, probs)
#"print(y)
def sample_beta(X, B, y, b, samples=1, burn_in=100, sample_every=1):
    kappa = y - 0.5
    B_inv = np.linalg.inv(B)
    n, dim = X.shape[0], X.shape[1]
    print("n", n)
    beta_hat = np.random.randn(dim) * 0.001
    omega = None
    
    l = np.zeros((samples, dim))
    for i in progressbar.progressbar(list(range(samples * sample_every + burn_in))):
        XTbeta = beta_hat @ X.T
        omega = random_polyagamma(z=XTbeta)
        
        omega_sum = np.sum(omega)
        omega = omega / omega_sum
        #Omega = np.diag(omega)
        #V_omega_inv = X.T @ Omega @ X + B_inv
        V_omega_inv = (X.T * omega) @ X + B_inv
        V_omega = np.linalg.inv(V_omega_inv)
        m_omega = V_omega @ (X.T @ kappa + B_inv @ b )
        
        
        beta_hat = np.random.multivariate_normal(m_omega, V_omega)
        
        if i >= burn_in and i % sample_every == 0:
            i_prime = (i - burn_in ) // sample_every
            l[i_prime] = beta_hat
            
    return l
        
        
b = np.zeros(DIM)
B = np.identity(DIM)
betas = sample_beta(X, B, y, b, burn_in=5000, samples=100, sample_every=10)


print(np.mean(betas, axis=0))