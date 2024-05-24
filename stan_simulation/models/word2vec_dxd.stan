data {
    int<lower=1> N; // data size
    int<lower=1> V; // vocab size
    int<lower=1> D; // embedding dim
    array[N] int<lower=1, upper=V> target_word;
    array[N] int<lower=1, upper=V> context_word;
    array[N] int<lower=0, upper=1> posneg_labels; // 1: positive or 0: negative sample
    matrix[D, D] fixed_context_matrix; // fixed top DxD matrix
}

parameters {
    matrix[V, D] word_vectors;
    matrix[(V-D), D] context_vectors_raw; // context vectors excluding the fixed part
}

transformed parameters {
    matrix[V, D] context_vectors;

    // Fill in the fixed part
    for (i in 1:D) {
        for (j in 1:D) {
            context_vectors[i, j] = fixed_context_matrix[i, j];
        }
    }

    // Fill in the remaining part
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
            context_vectors_raw[v-D, d] ~ normal(0, 1);
        }
    }
    for (v in 1:V) {
        for (d in 1:D) {
            word_vectors[v, d] ~ normal(0, 1);
        }
    }

    // likelihood
    for (n in 1:N) {
        real l_contribution = dot_product(word_vectors[target_word[n]], context_vectors[context_word[n]]);
        if (posneg_labels[n] == 1) {
            target += log_inv_logit(l_contribution);
        } else {
            target += log_inv_logit(-l_contribution);
        }
    }
}
