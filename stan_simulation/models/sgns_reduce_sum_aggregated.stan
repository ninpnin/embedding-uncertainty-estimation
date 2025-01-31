functions {
  real partial_log_likelihood(int start, int end, 
                              array[int] target_word, 
                              array[int] context_word, 
                              array[int] posneg_labels, 
                              array[int] counts, 
                              matrix word_vectors, 
                              matrix context_vectors) {
      real result = 0;
      for (u in start:end) {
          real l_contribution = dot_product(word_vectors[target_word[u]], context_vectors[context_word[u]]);
          if (posneg_labels[u] == 1) {
              result += counts[u] * log_inv_logit(l_contribution);
          } else {
              result += counts[u] * log_inv_logit(-l_contribution);
          }
      }
      return result;
  }
}

data {
  real<lower=0.0> lambda;
  int<lower=1> U; // unique pairs count
  int V; // vocab size
  int D; // embedding dim
  array[U] int target_word;
  array[U] int context_word;
  array[U] int<lower=0, upper=1> posneg_labels; // 1: positive or 0: negative sample
  array[U] int counts; // counts of each unique pair
}

parameters {
  matrix[V,D] word_vectors;
  matrix[V,D] context_vectors;
}

model {
  // prior
  to_vector(word_vectors) ~ normal(0, lambda);
  to_vector(context_vectors) ~ normal(0, lambda);

  // likelihood
  target += reduce_sum(partial_log_likelihood, 1, target_word, context_word, posneg_labels, counts, word_vectors, context_vectors);
}
