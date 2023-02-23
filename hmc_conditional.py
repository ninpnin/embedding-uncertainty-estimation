import tensorflow as tf
import numpy as np
import copy
import networkx as nx
import progressbar as pb

from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.models import cbow_likelihood
from probabilistic_word_embeddings.estimation import map_estimate

#  0. Find MAP embedding to obtain the context vectors

#  1. Preprocess data

#  2. Select words to be sampled, loop through data to find their 
#  associated context windows

#  3. Draw negative samples, keep them across draws

#  4. Run HMC
## a. randomize embeddings for the words in question
## b. randomize momentum from unit normal
## c. Calculate gradient for positive and negative samples
## d. Update momentum and embedding values for the words in question
## based on leapfrog integration
## e. accept or reject sample
## f. Repeat for b-e for eg. 4000 steps and keep the values

# 5. Create an pwe.Embedding instance for each (?) draw
# Set indices so that e["dog"] / e[["cat", "dog"]] style syntax
# can be used and returns a correctly structured array

def propose(wds, e, data, epsilon=0.01, leapfrog_steps=20):
    i, j, x = data
    
    e_gradient = copy.deepcopy(e)

    momentum = copy.deepcopy(e)
    location = copy.deepcopy(e)
    momentum[wds] = tf.random.normal(momentum[wds].shape, dtype=tf.float64)

    potential_0 = - tf.reduce_sum(cbow_likelihood(location, i, j, x=x))
    kinetic_0 = 0.5 * tf.reduce_sum(tf.multiply(momentum[wds] ,momentum[wds] ))
    print("Potential", potential_0.numpy(), "kinetic", kinetic_0.numpy())
    
    energy_0 = potential_0 + kinetic_0

    for step in pb.progressbar(range(leapfrog_steps)):
        with tf.GradientTape() as tape:
            objective = tf.reduce_sum(cbow_likelihood(location, i, j, x=x))
            gradient = - tape.gradient(objective, location.theta)
        e_gradient.theta.assign(gradient)
        
        momentum[wds] -= e_gradient[wds] * epsilon
        location[wds] += momentum[wds] * epsilon

    potential_1 = - tf.reduce_sum(cbow_likelihood(location, i, j, x=x))
    kinetic_1 = 0.5 * tf.reduce_sum(tf.multiply(momentum[wds], momentum[wds] ))
    print("Potential", potential_1.numpy(), "kinetic", kinetic_1.numpy())
    energy_1 = potential_1 + kinetic_1

    print(energy_0.numpy(), "->", energy_1.numpy())
    print(tf.exp(energy_1 - energy_0))

    ratio = tf.exp(energy_1 - energy_0)
    return location, ratio.numpy(), potential_0.numpy()

def conditional_hmc(wds, e, text, ws=5):
    # positive samples
    i = []
    j = []
    x = []

    for ix, wd in enumerate(text):
        if wd in wds:
            i.append(wd)
            context = text[ix-ws:ix] + text[ix+1:ix+ws+1]

            j.append(context)
            x.append(1)

    # negative samples
    for _ in range(len(i)):
        ix = np.random.randint(ws, len(text)-ws)
        context = text[ix-ws:ix] + text[ix+1:ix+ws+1]
        j.append(context)
        x.append(0)

    i = i + i

    i = tf.constant(i)
    j = tf.constant(j)
    x = tf.constant(x, dtype=tf.float64)
    print(i.shape, j.shape, x.shape)
    print(i.dtype, j.dtype, x.dtype)

    j = j + "_c"

    data = i,j, x
    lls = []
    e[wds] = tf.random.normal(e[wds].shape, dtype=tf.float64) / e[wds].shape[-1]
    for i in range(100):
        e_prime, ratio, ll = propose(wds, e, data, epsilon=0.001)

        print(ratio)
        if np.random.rand() < ratio:
            e[wds] = e_prime[wds]
            print("ACCEPT")
        else:
            print("REJECT")

        lls.append(ll)

    print(lls)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    text = open("wikismall.txt").read().lower().split()
    text, vocabulary = preprocess_standard(text)
    print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

    e = Embedding(saved_model_path="./lapl_emb.pkl")
    #e = Embedding(vocabulary=vocabulary, dimensionality=100)

    print(e)
    # Perform MAP estimation
    wds = ["dog"]

    e = conditional_hmc(wds, e, text)


