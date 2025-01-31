functions {
    real partial_sum(int[] posneg_labels_slice, int start, int end,
                     matrix word_vectors, matrix context_vectors, 
                     int[] target_word, int[] context_word) {
        real log_lik = 0;
        for (n in start:end) {
            vector[D] word_vec = word_vectors[target_word[n]];
            vector[D] context_vec = context_vectors[context_word[n]];
            

            // likelihood
            real l_contribution = dot_product(word_vec, context_vec);

            if (posneg_labels_slice[n] == 1) {
                log_lik += log_inv_logit(l_contribution);
            } else {
                log_lik += log_inv_logit(-l_contribution);
            }
        }
        return log_lik;
    }
}

data {
    real<lower=0.0> lambda;
    int N; // data size
    int V; // vocab size
    int D; // embedding dim
    array[N] int target_word; // indices for target words
    array[N] int context_word; // indices for context words
    array[N] int<lower=0, upper=1> posneg_labels; // 1: positive, 0: negative
}

parameters {
    matrix[V, D] word_vectors; 
    matrix[V, D] context_vectors; 
}

model {
    to_vector(word_vectors) ~ normal(0, lambda);
    to_vector(context_vectors) ~ normal(0, lambda);

    target += reduce_sum(partial_sum, posneg_labels, 1, word_vectors, context_vectors, target_word, context_word);
}
