// Fix so that the top D context embeddings form a triangular matrix. Enforcing d(d-1)/2 params are 0.
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
    matrix[V,D] context_vectors_raw;
}

// using spherical priors it is sufficient to fix d(d-1)/2
// fixate DxD top matrix to be upper triangular.
transformed parameters {
    matrix[V, D] context_vectors;

    for (v in 1:V) {
        for (d in 1:D) {
            if (v <= D && v >= d) {
                context_vectors[v, d] = context_vectors_raw[v, d];
            } else if (v <= D && v < d) {
                context_vectors[v, d] = 0;
            } else {
                context_vectors[v, d] = context_vectors_raw[v, d];
            }
        }
    }
}

model {
    // priors. Only apply prior to non fix part..
    for (v in 1:V) {
        for (d in 1:D) {
            if (v > D || v <= d) {
                context_vectors_raw[v, d] ~ normal(0, 1);
            }
            word_vectors[v, d] ~ normal(0, 1);
        }
    }

    // likelihood
    // todo: possibly aggregate
    for (n in 1:N) {
        real l_contribution = dot_product(word_vectors[target_word[n]], context_vectors[context_word[n]]);
        if (posneg_labels[n] == 1) {
            target += log_inv_logit(l_contribution);
        } else {
            target += log_inv_logit(-l_contribution);
        }
    }
}