library(bsda)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(rstan)
library(shinystan)
# Warmup: Eight schools
setwd(dir = "~/Work/embedding-uncertainty-estimation/tests")
X <- c(1.5, -0.39, 1.77)
print(X)
Sigma <- matrix(c(1,0,0,1),ncol=2)
biosassay_dat <- list(k=3, x=X, y=c(1,0,1), n=c(1,1,1), mu=c(0,0), Sigma=Sigma)
set.seed(123)
biosassay_fit <- stan(file = 'bioassay_model.stan', data = biosassay_dat, iter=32000, chains=32)

## Task 2. analyze convergence with R hat
biosassay_fit
alphas <- extract(biosassay_fit, pars="alpha", permute=FALSE)
mean(alphas)
sqrt(var(alphas)/81553)
Rhat(as.matrix(alphas))
betas <- extract(biosassay_fit, pars="beta", permute=FALSE)
mean(betas)
sqrt(var(betas)/81553)
stan_get(stanfit, what = "n_eff")

