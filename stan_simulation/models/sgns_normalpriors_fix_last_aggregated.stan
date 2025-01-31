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
    for (v in 1:V) {
        for (d in 1:D) {
            word_vectors[v, d] ~ normal(0,lambda);
            context_vectors[v,d] ~ normal(0,lambda);
        }
    } 

    // likelihood
    for (u in 1:U) {
        real l_contribution = dot_product(word_vectors[target_word[u]], context_vectors[context_word[u]]);
        if (posneg_labels[u] == 1) {
            target += counts[u] * log_inv_logit(l_contribution);
        } else {
            target += counts[u] * log_inv_logit(-l_contribution);
        }
    }
}
