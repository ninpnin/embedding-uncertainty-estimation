setwd("~/Work/embedding-uncertainty-estimation")
library("rstan")
options(mc.cores=4)

# Simulating some data
D <- 2
M <- 10
N <- 5
v <- c(1,4,2,4,9)
w <- c(3,5,6,7,8)
x <- c(1,1,1,0,0)

# Running stan code
model = stan_model("stan-models/bernoulli_embeddings.stan")

fit = sampling(model,list(D=D, M=M, N=N, v=v, w=w, x=x, sigma=1.0),iter=500,chains=4)

print(fit)

params = extract(fit)

par(mfrow=c(1,2))
ts.plot(params$mu,xlab="Iterations",ylab="mu")
hist(params$sigma,main="",xlab="sigma")