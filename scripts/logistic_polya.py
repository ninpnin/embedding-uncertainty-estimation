from trainerlog import get_logger
LOGGER = get_logger("sampling")
LOGGER.info("Load modules..")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from polyagamma import random_polyagamma

def get_v_omega(X, omega, sigma_prior):
    Omega = np.diag(omega)
    V_inv = X.T @ Omega @ X
    V_inv += np.linalg.inv(sigma_prior)
    return np.linalg.inv(V_inv)

def get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega):
    kappa = y - N/2
    parenthesis = X.T @ kappa + np.linalg.inv(sigma_prior) @ mu_prior
    return V_omega @ parenthesis

def polyagamma_gibbs(beta_init, X, y, iterations=2, N=None, mu_prior=None, sigma_prior=None):
    if N is None:
        N = np.ones(len(X))
    beta = beta_init
    
    if mu_prior is None:
        mu_prior = np.zeros(X.shape[-1])
    if sigma_prior is None:
        sigma_prior = np.identity(X.shape[-1])
        
    for _ in range(iterations):
        # Get a 5 by 1 array of PG(1, 2) variates.
        xTbeta = X @ beta
        omega = random_polyagamma(N, xTbeta)
        #print(omega)
        V_omega = get_v_omega(X, omega, sigma_prior)
        mu_omega = get_mu_omega(X, y, N, mu_prior, sigma_prior, V_omega)
        
        beta = np.random.multivariate_normal(mean=mu_omega, cov=V_omega)
        yield beta
        
        
LOGGER.train("Load data")
bioassay = pd.read_csv("data/bioassay.csv")
print(bioassay)

#X = np.random.rand(3,2)
#y = np.array([1,0,1])
X = np.array([bioassay["x"], np.ones(len(bioassay))]).T
y = np.array(bioassay["y"])
N = np.array(bioassay["n"])
sigma_prior = np.identity(2) * 100.0
print(X)
print(len(X))
beta_init = np.random.randn(2) * 0.2
LOGGER.train("Start sampling")
rows = [x for x in polyagamma_gibbs(beta_init, X, y, N=N, sigma_prior=sigma_prior, iterations=100000)]
LOGGER.train("Done!")
df = pd.DataFrame(rows, columns=["beta1", "beta2"])
df["method"] = "sequential"
df = df.head(500)

index = 50
LOGGER.train(f"Take the {index}th sample of multiple chains..")
rows = [[x for x in polyagamma_gibbs(beta_init, X, y, N=N, sigma_prior=sigma_prior, iterations=50)][-1] for _ in range(500)]
LOGGER.train("Done!")
df2 = pd.DataFrame(rows, columns=["beta1", "beta2"])
df2["method"] = f"one ({index}th iteration)"

df = pd.concat([df, df2])

sns.scatterplot(df, x="beta1", y="beta2", hue="method")
plt.show()

#plt.clf()
#sns.scatterplot(df.tail(100), x="beta1", y="beta2")
#plt.show()