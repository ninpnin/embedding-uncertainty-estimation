from probabilistic_word_embeddings.embeddings import Embedding
import numpy as np
import tensorflow as tf
import copy

def get_subhessian(alpha, rho, x=1):
    if x == 0:
        x = -1
    alpha = tf.Variable(alpha)
    rho = tf.Variable(rho)
    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        eta = tf.reduce_sum(x * alpha * rho)
        p = tf.math.sigmoid(eta)
        log_p = tf.math.log(p)

      g = t1.gradient(log_p, alpha)
    return t2.jacobian(g, rho)


def get_laplace_hessian(e, data):
    vocabulary = e.vocabulary

    lambda0 = e.lambda0
    print(lambda0)
    print(vocabulary)
    dim = e[list(e.vocabulary)[0]].shape[0]
    print(dim)

    hessian_vocabulary = {key: val * dim for key, val in vocabulary.items()}
    print(hessian_vocabulary)

    H = np.zeros((dim * len(vocabulary), dim * len(vocabulary)))
    print(H)
    print(H.shape)

    H += np.identity(dim * len(vocabulary)) * lambda0
    print(H)

    for w, v, x in data:
        rho = e[w]
        alpha = e[v]
        subhessian = get_subhessian(alpha, rho, x=x)
        
        i, j = hessian_vocabulary[w], hessian_vocabulary[v]
        for m in range(0, dim):
            for n in range(0, dim):
                entry = subhessian[m,n]
                H[i + m, j + n] += entry
                H[j + n, i + m] += entry

    return H, hessian_vocabulary

def sample(e, L, hessian_vocabulary):
    epsilon_prime = np.random.randn(L.shape[0])
    epsilon = L @ epsilon_prime
    dim = e[list(e.vocabulary)[0]].shape[0]

    e_sample = copy.deepcopy(e)

    for v in e.vocabulary:
        i = hessian_vocabulary[v]
        e_sample[v] += epsilon[i: i+dim]
    
    return e_sample

if __name__ == '__main__':
    e = Embedding(saved_model_path="example_embedding.pkl")
    print(e)
    # {'joo': 0, 'moi': 1, 'jee': 2, 'joo_c': 3, 'moi_c': 5, 'jee_c': 4}
    data = [('joo', 'jee_c', 0), ('moi', 'joo_c', 1)]
    H, hessian_vocabulary = get_laplace_hessian(e, data)

    print(H)

    Sigma = np.linalg.inv(H)
    print(Sigma)

    L = np.linalg.cholesky(Sigma)

    print(L)

    print(e["joo"])
    for _ in range(10):
        e_sample = sample(e, L, hessian_vocabulary)
        print(e_sample["joo"])

