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
D <- 3
M <- max(v)
N <- length(v)

# Running stan code
model = stan_model("stan-models/bernoulli_embeddings.stan")
fit = sampling(model,list(D=D, M=M, N=N, v=v, w=w, x=x, sigma=2.0),iter=1000,chains=2)

model = stan_model("stan-models/bernoulli_embeddings_fixed.stan")
fit = sampling(model,list(D=D, M=M, N=N, v=v, w=w, x=x, sigma=2.0,fixed_indices=c(1,2)),iter=3000,chains=4)

print(fit)

# calculate true co-occurence matrix
p_ones = matrix(0, M, M)
p_zeros = matrix(0, M, M)
for (i in 1:N) {
  if (x[i] == 1) {
    p_ones[v[i], w[i]] <- p_ones[v[i], w[i]] + 1
  } else {
    p_zeros[v[i], w[i]] <- p_zeros[v[i], w[i]] + 1
  }
}
p_true <- p_ones / (p_ones + p_zeros)

p_true
params = extract(fit)
p_est <- apply(params$p, 2:3, mean)
cor(c(p_true), c(p_est),use='complete.obs')
