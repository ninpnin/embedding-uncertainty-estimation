data {
  int<lower=0> D;
  int<lower=0> J;
  real<lower=0.0> a;
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
  for (j in 1:J) {
    if (x[j] == 0){
      target += a * bernoulli_logit_lpmf(x[j] | eta[j]);
    } else {
      target += bernoulli_logit_lpmf(x[j] | eta[j]);
    }
  }
  target += normal_lpdf(rho| 0, 1);
}
