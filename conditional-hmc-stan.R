library(reticulate)
library(rjson)
library(rstan)

wd <- "cat"
NS <- 5
setwd("~/Work/embedding-uncertainty-estimation")

np <- import("numpy")
i <- np$load("data/stan/i.npy")
e_j <- np$load("data/stan/e_j.npy")
e_j_neg <- np$load("data/stan/e_j_neg.npy")
dim(e_j)
dim(e_j_neg)

freq <- fromJSON(file = "data/stan/freq.json")
freq_prime <- fromJSON(file = "data/stan/freq_prime.json")
freq_prime[wd]

l_w <- dim(e_j)[1]
l_ns <- dim(e_j_neg)[1]
a_dog <- NS * get(wd, freq_prime) / get(wd, freq) * l_w / l_ns
a_dog

x <- c(rep(c(1), each=l_w), rep(c(0), each=l_ns))
x

c_j <- rbind(e_j, e_j_neg)
c_j
dim(c_j)
train_dat <- list(x=x, D=dim(c_j)[2], J=dim(c_j)[1], c_j=c_j, a=a_dog)
fit1 <- stan(file = "bernoulli_embeddings.stan", data=train_dat, chains=2)
fit1

params = extract(fit1)
dim(params)
rho <-params$rho
dim(rho)
embedding_path <- paste("trained/rho_", wd, ".npy", sep = "")
embedding_path
np$save(embedding_path, rho)
