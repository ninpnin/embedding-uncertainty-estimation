data {
    real<lower=0.0> lambda; // prior strength
    int<lower=1> U; // unique pairs count
    int<lower=1> V; // vocab size
    int<lower=1> D; // embedding dim
    array[U] int target_word;
    array[U] int context_word;
    array[U] int<lower=0, upper=1> posneg_labels; // 1: positive or 0: negative sample
    array[U] int counts; // counts of each unique pair
    matrix[D, D] fixed_context_matrix; // fixed top D context vectors.
}

parameters {
    matrix[V, D] word_vectors;
    matrix[(V-D), D] context_vectors_raw; // context vectors excluding the fixed part
}

transformed parameters {
    matrix[V, D] context_vectors;

    for (i in 1:D) {
        for (j in 1:D) {
            context_vectors[i, j] = fixed_context_matrix[i, j];
        }
    }

    for (v in (D+1):V) {
        for (d in 1:D) {
            context_vectors[v, d] = context_vectors_raw[v-D, d];
        }
    }
}

model {
    // priors. Only apply prior to the non-fixed part
    for (v in (D+1):V) {
        for (d in 1:D) {
            context_vectors_raw[v-D, d] ~ normal(0, lambda);
        }
    }
    for (v in 1:V) {
        for (d in 1:D) {
            word_vectors[v, d] ~ normal(0, lambda);
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
