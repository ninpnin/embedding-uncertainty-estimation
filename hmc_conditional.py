import tensorflow as tf
import numpy as np

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

