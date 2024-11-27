data {
  int<lower=0> k;
  vector[k] x;
  int<lower=0> y[k];
  int<lower=0> n[k];
  vector[2] mu;
  matrix[2,2] Sigma;
}
parameters {
  vector[2] theta;
}
transformed parameters {
  real alpha = theta[1];
  real beta = theta[2];
}
model {
  theta ~ multi_normal(mu, Sigma);
  y ~ binomial_logit(n, alpha + beta * x);
}
