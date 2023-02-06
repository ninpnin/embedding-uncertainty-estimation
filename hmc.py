import networkx as nx
from probabilistic_word_embeddings.preprocessing import preprocess_standard
import tensorflow as tf
# Load model from models.py
from probabilistic_word_embeddings.utils import shuffled_indices
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.models import generate_sgns_batch, sgns_likelihood
from probabilistic_word_embeddings.models import generate_cbow_batch, cbow_likelihood
from probabilistic_word_embeddings.evaluation import evaluate_word_similarity, evaluate_on_holdout_set

import progressbar
from scipy.spatial.distance import cosine as cos_dist
import random

def hmc(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, epochs=5, learning_rate=0.01, evaluate=False):
    learning_rate = tf.constant(learning_rate, dtype=tf.float64)
    if not isinstance(embedding, Embedding):
        warnings.warn("embedding is not a subclass of probabilistic_word_embeddings.Embedding")
    if model not in ["sgns", "cbow"]:
        raise ValueError("model must be 'sgns' or 'cbow'")

    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)

    e = embedding
    N = len(data)
    batches = N // batch_size

    embedding.theta.assign(tf.random.normal(embedding.theta.shape, dtype=tf.float64))
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)
            wa = sum(similarity["Rank Correlation"] * similarity["No. of Observations"]) / sum(similarity["No. of Observations"])
        
            print("Weighted average", wa)

        r = tf.random.normal(embedding.theta.shape, dtype=tf.float64) / batches
        r = tf.Variable(r, name="momentum")

        speedbreak = 1.0
        # Shuffle the order of batches
        for batch in progressbar.progressbar(random.sample(range(batches),batches)):
            start_ix = batch_size * batch
            with tf.GradientTape() as tape:
                if model == "sgns":
                    i,j,x  = generate_sgns_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
                    objective = - tf.reduce_sum(sgns_likelihood(embedding, i, j, x=x)) - embedding.log_prob(batch_size, N)
                elif model == "cbow":
                    i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
                    objective = - tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
            gradient = tape.gradient(objective, embedding.theta)
            speedbreak = 1.0 / (1.0 + 0.1 * tf.reduce_max(tf.math.abs(r)))
            #gradient = gradient
            r.assign_sub(gradient * learning_rate * speedbreak)
            print(r)
            print(embedding.theta)
            embedding.theta.assign_add(r * learning_rate * speedbreak)

    return embedding


text = open("wiki.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

g = nx.Graph()
g.add_edge("this", "that")
dim = 100
e = LaplacianEmbedding(vocabulary, dim, g)
# Perform MAP estimation

e = hmc(e, text, model="cbow", ws=5, epochs=15, learning_rate=0.01)
similarity = evaluate_word_similarity(e)
print(similarity)