data {
  real<lower=0> s;     // N(0, s²) prior std. (In simulation experiment : s^2 = 1/sqrt(λ))
  int<lower=1> V, K, N, WS;                        // vocab, dim, observations, context window size

  array[N] int<lower=1, upper=V> target_word;      
  array[N, WS] int<lower=1, upper=V> context_idx;  // context indices per target word
  array[N] int<lower=0, upper=1> posneg_labels;    // 0/1 labels
}
parameters {
  matrix[V, K] rho;
  matrix[V, K] alpha;
}
model {
  to_vector(rho)  ~ normal(0, s);
  to_vector(alpha) ~ normal(0, s);

  // Likelihood
  for (n in 1:N) {
    row_vector[K] sum_ctx = rep_row_vector(0, K);
    for (j in 1:WS) {
      int v = context_idx[n,j];
      sum_ctx += alpha[v, ];
    }
    posneg_labels[n] ~ bernoulli_logit(dot_product(rho[target_word[n], ], sum_ctx));
  }
}
