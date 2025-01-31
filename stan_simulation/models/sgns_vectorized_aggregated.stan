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
    vector[U] l_contributions;
    for (u in 1:U) {
        l_contributions[u] = dot_product(word_vectors[target_word[u]], context_vectors[context_word[u]]);
    }
    vector[U] logits = to_vector(counts) .* (to_vector(posneg_labels) .* log_inv_logit(l_contributions) +
                                             (1 - to_vector(posneg_labels)) .* log_inv_logit(-l_contributions));
    target += sum(logits);
}
