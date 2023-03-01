library(reticulate)
library(rjson)
library(rstan)

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
freq_prime["dog"]

l_w <- dim(e_j)[1]
l_ns <- dim(e_j_neg)[1]
a_dog <- NS * get("dog", freq_prime) / get("dog", freq) * l_w / l_ns
a_dog

x <- c(rep(c(1), each=l_w), rep(c(0), each=l_ns))


c_j <- rbind(e_j, e_j_neg)
dim(c_j)
train_dat <- list(x=x, D=dim(c_j)[2], J=dim(c_j)[1], c_j=c_j)
fit1 <- stan(file = "bernoulli_embeddings.stan", data=train_dat)
