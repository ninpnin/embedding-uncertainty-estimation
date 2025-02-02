data {
    real<lower=0.0> lambda;             // prior strength
    int<lower=1> U;                     // unique pairs count
    int<lower=1> V;                     // vocabulary size
    int<lower=1> D;                     // embedding dimension
    array[U] int target_word;
    array[U] int context_word;
    array[U] int<lower=0, upper=1> posneg_labels; // 1 => positive or 0 => negative sample
    array[U] int counts;                // counts of each unique pair
    
    matrix[D, D] fixed_context_matrix;
}

parameters {
    matrix[V, D] word_vectors;
    
    matrix[(V - D), D] context_vectors_raw; 
}

transformed parameters {
    matrix[V, D] context_vectors;

    for (v in 1:(V - D)) {
        for (d in 1:D) {
            context_vectors[v, d] = context_vectors_raw[v, d];
        }
    }
    for (row_index in 1:D) {
        for (col_index in 1:D) {
            context_vectors[V-D + row_index, col_index] = fixed_context_matrix[row_index, col_index];
        }
    }
}

model {

    for (v in 1:(V-D)) {
        for (d in 1:D) {
            context_vectors_raw[v, d] ~ normal(0, lambda);
        }
    }
    // Priors for the word vectors
    for (v in 1:V) {
        for (d in 1:D) {
            word_vectors[v, d] ~ normal(0, lambda);
        }
    }

    // Likelihood
    for (u in 1:U) {
        real l_contribution = dot_product(
            word_vectors[target_word[u]], 
            context_vectors[context_word[u]]
        );
        if (posneg_labels[u] == 1) {
            target += counts[u] * log_inv_logit(l_contribution);
        } else {
            target += counts[u] * log_inv_logit(-l_contribution);
        }
    }
}
