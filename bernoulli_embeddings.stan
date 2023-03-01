data {
  int<lower=0> D;
  int<lower=0> J;
  int<lower=0, upper=1> x[J];
  matrix[J, D] c_j;
}

parameters {
  vector[D] rho;
}

transformed parameters {
  vector[J] eta = c_j * rho;
}

model {
  target += bernoulli_logit_lpmf(x | eta);
  target += normal_lpdf(rho| 0, 1);
}