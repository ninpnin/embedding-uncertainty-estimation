from trainerlog import get_logger
LOGGER = get_logger("sampling")
LOGGER.info("Load modules..")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from embedding_uncertainty import polyagamma_sampler
        
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

print("X.shape, y.shape, beta_init.shape", X.shape, y.shape, beta_init.shape)
rows = [x for x in polyagamma_sampler(beta_init, X, y, N=N, sigma_prior=sigma_prior, iterations=100000)]
LOGGER.train("Done!")
df = pd.DataFrame(rows, columns=["beta1", "beta2"])
df["method"] = "sequential"
df = df.head(500)

index = 50
LOGGER.train(f"Take the {index}th sample of multiple chains..")
rows = [[x for x in polyagamma_sampler(beta_init, X, y, N=N, sigma_prior=sigma_prior, iterations=50)][-1] for _ in range(500)]
LOGGER.train("Done!")
df2 = pd.DataFrame(rows, columns=["beta1", "beta2"])
df2["method"] = f"one ({index}th iteration)"

df = pd.concat([df, df2])

sns.scatterplot(df, x="beta1", y="beta2", hue="method")
plt.show()

#plt.clf()
#sns.scatterplot(df.tail(100), x="beta1", y="beta2")
#plt.show()