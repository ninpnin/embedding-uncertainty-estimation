import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy
import networkx as nx
import progressbar as pb

from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.models import cbow_likelihood
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.utils import transfer_embeddings
import random

def generate_dataset(e, batches, batch_size):
    words = [wd for wd in list(e.vocabulary) if "_c" not in wd]

    for batch in range(batches):
        i = random.choices(words, k=batch_size)
        j = random.choices(words, k=batch_size)
        j = [wd + "_c" for wd in j]
        print(i)
        print(j)

        e_i, e_j = e[i], e[j]

        etas = tf.math.sigmoid(tf.reduce_sum(tf.multiply(e_i, e_j), axis=1))
        print(etas)
        x = tfp.distributions.Bernoulli(probs=etas).sample()
        print(x)

        i_j_x = (i,j,x)
        yield i_j_x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    e = Embedding(saved_model_path="trained/lapl_emb.pkl")

    words = ["this", "will", "about", "food", "democrats", "republicans"]
    e_prime = Embedding(set(words), dimensionality=100)
    e_prime = transfer_embeddings(e, e_prime)

    data = list(generate_dataset(e_prime, 1, 7))
    print(data)


