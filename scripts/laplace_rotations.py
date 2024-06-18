import tensorflow as tf
import numpy as np
from probabilistic_word_embeddings.embeddings import Embedding


def _sample_rotation_tangent(e):
  def _sample_S(dim):
    S = np.random.randn(dim)
    S = np.triu(S)
    S = S - S.T
    return S

  dim = e.theta.shape[1]
  S = _sample_S(dim)
  v = e.theta @ S

  return v.numpy().flatten()

def remove_rotations(e):
  dim = e.theta.shape[1]
  vs = [_sample_rotation_tangent(e) for _ in range(dim * (dim-1) // 2)]
  vs = np.array(vs)

  q, _ = np.linalg.qr(vs.T)

  return q


dim = 3
words = ["moi", "mitä", "sinulle", "kuuluu", "joo", "ei"]

e = Embedding(set(words), dimensionality=dim)
print(e)

q = remove_rotations(e)
print(q)
