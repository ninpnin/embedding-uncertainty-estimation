data {
  int<lower=0> D; // Embedding dimensionality
  int<lower=0> M; // Vocabulary size
  int<lower=0> N; // Number of data points
  
  int<lower=1, upper=M> v[N]; // center words
  int<lower=1, upper=M> w[N]; // context words
  int<lower=0, upper=1> x[N]; // bernoulli outcomes
  
  real<lower=0.0> sigma; // prior strength

  int<lower=1, upper=M> fixed_indices[D]; // 
}

parameters {
  matrix[M, D] rho;
  matrix[M, D] alpha;
}

model {
  for (j in 1:N) {
    int v_j = v[j];
    int w_j = w[j];
    
    real eta_j = dot_product(rho[v_j], alpha[w_j]);
    for (ix in 1:D) {
      if (fixed_indices[ix] == w_j) {
        //print("w: ", w_j);
        for (ix2 in 1:ix) {
          //print("ix2: ", ix2);
          eta_j = dot_product(rho[v_j, 1:ix2], alpha[w_j, 1:ix2]);
        }
      }
    }


    target += bernoulli_logit_lpmf(x[j] | eta_j);
  }
  for (i in 1:M) {
    for (d in 1:D) {
      target += normal_lpdf(rho[i,d]   | 0.0, sigma);
      target += normal_lpdf(alpha[i,d] | 0.0, sigma);
    }
  }
}

generated quantities {
  matrix[M,M] p;
  for (i in 1:M) {
    for (j in 1:M) {
      p[i,j] = inv_logit(dot_product(rho[i], alpha[j]));
    }
  }
}
