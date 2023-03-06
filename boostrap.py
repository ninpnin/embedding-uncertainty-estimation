from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
import tensorflow as tf
import numpy as np
from pathlib import Path

bootstrap_resamples = 10
resample_len = 500
dim = 100
epochs = 15
text = open("data/wiki.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

datapoints = len(text) // resample_len
print(datapoints)
datapoints = list(range(datapoints))

trained_model_folder = Path("trained")
if not trained_model_folder.exists():
    trained_model_folder.mkdir()
    
for r in range(bootstrap_resamples):
    # Resampling indices
    r_indices = np.random.choice(datapoints, size=len(datapoints))
    
    # Resampled data
    text_r = []
    for ix in r_indices:
        start, end = ix * resample_len, (ix + 1) * resample_len
        text_r += text[start:end]
    
    # Check that everything looks alright
    text_r = tf.constant(text_r)
    print(text_r)
    
    # Train an embedding on the resampled data
    e = Embedding(vocabulary, dim)
    e = map_estimate(e, text_r, model="cbow", ws=5, epochs=epochs)
    
    # Save embedding
    resample_path = trained_model_folder / f"boostrap_{r}.pkl"
    e.save(resample_path.absolute())