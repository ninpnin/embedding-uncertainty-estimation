setwd("~/Work/embedding-uncertainty-estimation")
library("rstan")
options(mc.cores=4)
library("rjson")
myData <- fromJSON(file="example.json")

v <- c()
w <- c()
x <- c()

for (p in myData$data) {
  #print(p)
  v_i <- p$v
  w_i <- p$w
  x_i <- p$x
  
  # Extract index "2" from word "word_2"
  v_i <- substr(v_i, nchar(v_i), nchar(v_i))
  w_i <- substr(w_i, nchar(w_i), nchar(w_i))
  
  # Convert index "2" to integer,
  # and add 1 due to one-based indexing
  v_i <- strtoi(v_i) + 1
  w_i <- strtoi(w_i) + 1
  
  v = c(v, c(v_i))
  w = c(w, c(w_i))
  x = c(x, c(x_i))
  
}
print(v)
# Simulating some data
D <- 2
M <- max(v)
N <- length(v)

# Running stan code
model = stan_model("stan-models/bernoulli_embeddings.stan")

fit = sampling(model,list(D=D, M=M, N=N, v=v, w=w, x=x, sigma=2.0),iter=500,chains=4)

print(fit)

params = extract(fit)

par(mfrow=c(1,2))
ts.plot(params$mu,xlab="Iterations",ylab="mu")
hist(params$sigma,main="",xlab="sigma")