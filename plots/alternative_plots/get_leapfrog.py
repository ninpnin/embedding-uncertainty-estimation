import pickle, numpy as np

with open("stan_fit_cmd_hmc_K50_congress-stemmed-ws-2-ns-1-vocab-5000-train_50000.pkl", "rb") as f:
    fit = pickle.load(f)

lf = fit.method_variables()["n_leapfrog__"]
print(lf)
print(np.shape(lf))
np.save("leapfrogs.npy", lf)
print(lf.mean(), lf.max())

if True:
    try:
        config = fit.metadata.cmdstan_config
        #print({k: v for k, v in config.items() if any(x in k.lower() for x in ["time", "elapsed", "warm", "sample", "iter"])})
        print(config.keys())
    except:
        print('Error')
        pass

print(fit.summary())