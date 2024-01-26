// Bernoulli embeddings SGNS / CBOW model, so that the estimation is conditional
// on trained context vectors. These trained context vectors are provided as 
// numerical values in the c_j variable
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

// Since the context vectors are set in stone, there is no difference between 
// CBOW and SGNS here
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
