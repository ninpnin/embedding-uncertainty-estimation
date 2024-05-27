data {
    int N; // data size
    int V; // vocab size
    int D; // embedding dim
    array[N] int target_word;
    array[N] int context_word;
    array[N] int<lower=0, upper=1> posneg_labels; // 1: positive or 0: negative sample
}


parameters {
    matrix[V,D] word_vectors;
    matrix[V,D] context_vectors;
}

model {
    // prior
    for (v in 1:V) {
        for (d in 1:D) {
            word_vectors[v, d] ~ normal(0,1);
            context_vectors[v,d] ~ normal(0,1);
        }
    } 

    // likelihood.
    // "target is a special variable representing the log-probability accumulator. Adding to target means adding to log-likelihood of the model."
    for (n in 1:N) {
        real l_contribution = dot_product(word_vectors[target_word[n]], context_vectors[context_word[n]]);
        if (posneg_labels[n] == 1) {
            target += log_inv_logit(l_contribution); //"log_inv_logit returns the natural logarithm of the inverse logit of the specified argument". inverse logit = sigmoid
        } else {
            target += log_inv_logit(-l_contribution);
        }
    }
}
