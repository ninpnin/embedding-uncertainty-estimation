from trainerlog import get_logger
LOGGER = get_logger("laplace", splitsec=True)
LOGGER.info("Load modules..")
import tensorflow as tf
import numpy as np
import copy
import bidict
LOGGER.info("Done!")
from probabilistic_word_embeddings.models import sgns_likelihood

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def aggregate_data(data):
  LOGGER.info(f"Aggregate data ...")
  pos_samples = {}
  neg_samples = {}

  for item in data:
    i, j, x = item
    pair = (i,j)
    if x == 1.0:
      pos_samples[pair] = pos_samples.get(pair, 0) + 1
    elif x == 0.0:
      neg_samples[pair] = neg_samples.get(pair, 0) + 1
    else:
      LOGGER.error(f"Wut, x was {x}")
      return
  return pos_samples, neg_samples

def gradient(e, data):
  """
  Off-diagonal subhessian
  """
  e_grad = copy.deepcopy(e)
  pos_samples, neg_samples = aggregate_data(data)

  words = [wd for wd in e.vocabulary if "_c" not in wd]
  contexts = [wd for wd in e.vocabulary if "_c" in wd]

  e_grad[words] = e_grad[words] * 0.0
  e_grad[contexts] = e_grad[contexts] * 0.0

  LOGGER.info(f"Loop over {len(pos_samples)} and {len(neg_samples)} samples ...")
  for w in words:
    rho = e[w].numpy()
    for v in contexts:
      pair = (w, v)
      if pair in pos_samples or pair in neg_samples:
        alpha = e[v].numpy()
        n_plus = pos_samples.get(pair, 0)
        n_minus = neg_samples.get(pair, 0)
        
        eta = sigmoid(np.dot(rho, alpha))
        multiplier = (n_plus + n_minus) * (n_plus / (n_plus + n_minus) - eta)
        e_grad[w] = e_grad[w] + multiplier * alpha
        e_grad[v] = e_grad[v] + multiplier * rho

  for w in words:
    e_grad[w] = e[w] * e.lambda0
  for v in contexts:
    e_grad[v] = e[v] * e.lambda0

  return e_grad

def gradient_tf(e, data):
  """
  Off-diagonal subhessian
  """
  e_grad = copy.deepcopy(e)
  i_batch = tf.constant([i for i,j,x in data])
  j_batch = tf.constant([j for i,j,x in data])
  x_batch = tf.constant([x for i,j,x in data], dtype=tf.float64)
  i, j, x = i_batch, j_batch, x_batch
  N = len(i)
  batch_size = N
  
  with tf.GradientTape() as t1:
    objective = - tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
    g = t1.gradient(objective, e.theta)

  e_grad.theta.assign(g)
  return e_grad


def hessian_tf(e, data, V, K):
  """
  Off-diagonal subhessian
  """
  e_grad = copy.deepcopy(e)
  i_batch = tf.constant([i for i,j,x in data])
  j_batch = tf.constant([j for i,j,x in data])

  x_batch = tf.constant([x for i,j,x in data], dtype=tf.float64)
  i, j, x = i_batch, j_batch, x_batch
  N = len(i)
  batch_size = N
  
  with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
      objective = - tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
      g = t1.gradient(objective, e.theta)

    hessian = t2.jacobian(g, e.theta)

  hessian = tf.reshape(hessian, [2*V*K, 2*V*K])
  return hessian

def subhessian(n_plus, n_minus, rho, alpha, i=0, j=1):
    """
    Off-diagonal subhessian
    """
    d = len(rho)
    rho = tf.Variable(rho)
    alpha = tf.Variable(alpha)
    variables = [rho, alpha]

    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        eta_plus = tf.sigmoid(tf.reduce_sum(tf.multiply(rho, alpha)))
        loss_plus = n_plus * tf.math.log(eta_plus)

        eta_minus = tf.sigmoid(- tf.reduce_sum(tf.multiply(rho, alpha)))
        loss_minus = n_minus * tf.math.log(eta_minus)

        loss = - loss_plus - loss_minus

      g = t1.gradient(loss, variables[i])

    h = t2.jacobian(g, variables[j])

    return h

def subhessian_analytic(n_plus, n_minus, rho, alpha, i=0, j=1):
    """
    Off-diagonal subhessian, calculate directly from the derivations
    """
    K = len(rho)
    dotprod = np.dot(alpha, rho)
    eta = tf.sigmoid(dotprod)
    eta_minus = 1.0 - tf.sigmoid(dotprod)
    logs_second_d = eta * eta_minus

    if i != j:
      raT = np.outer(rho, alpha).T
      h = raT * logs_second_d * (n_plus + n_minus)
      h_pos = - np.identity(K) * eta_minus
      h_neg = np.identity(K) * eta
      if i == 0 and j == 1:
        return h + h_pos * n_plus + h_neg * n_minus
      else:
        return (h + h_pos * n_plus + h_neg * n_minus).T
    elif i == 0 and j == 0:
      aaT = np.outer(alpha, alpha).T
      return aaT * logs_second_d * (n_plus + n_minus)
    elif i == 1 and j == 1:
      rrT = np.outer(rho, rho).T
      return rrT * logs_second_d * (n_plus + n_minus)



def subhessian(n_plus, n_minus, rho, alpha, i=0, j=1):
    """
    Diagonal subhessian
    """
    d = len(rho)
    rho = tf.Variable(rho)
    alpha = tf.Variable(alpha)
    variables = [rho, alpha]

    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        eta_plus = tf.sigmoid(tf.reduce_sum(tf.multiply(rho, alpha)))
        loss_plus = n_plus * tf.math.log(eta_plus)

        eta_minus = tf.sigmoid(- tf.reduce_sum(tf.multiply(rho, alpha)))
        loss_minus = n_minus * tf.math.log(eta_minus)

        loss = - loss_plus - loss_minus

      g = t1.gradient(loss, variables[i])

    h = t2.jacobian(g, variables[j])

    return h


def full_hessian(e, data):
  K = e.dimensionality
  words = [wd for wd in e.vocabulary if "_c" not in wd]
  contexts = [wd for wd in e.vocabulary if "_c" in wd]
  V = len(words)
  full_H = np.zeros((V * K * 2, V * K * 2))

  pos_samples, neg_samples = aggregate_data(data)

  LOGGER.info(f"Loop over {len(pos_samples)} and {len(neg_samples)} samples ...")
  for w in words:
    rho = e[w].numpy()
    for v in contexts:
      pair = (w, v)
      if pair in pos_samples or pair in neg_samples:
        alpha = e[v].numpy()
        n_plus = pos_samples.get(pair, 0)
        n_minus = neg_samples.get(pair, 0)

        H_ww = subhessian_analytic(n_plus, n_minus, rho, alpha, i=0, j=0)
        H_vv = subhessian_analytic(n_plus, n_minus, rho, alpha, i=1, j=1)
        H_wv = subhessian_analytic(n_plus, n_minus, rho, alpha, i=0, j=1)
        H_vw = H_wv.numpy().T

        i, j = e.vocabulary[w], e.vocabulary[v]
        i_ix = i * K
        j_ix = j * K

        # Diagonal
        full_H[i_ix:i_ix+K, i_ix:i_ix+K] += H_ww
        full_H[j_ix:j_ix+K, j_ix:j_ix+K] += H_vv

        # Off-diagonal
        full_H[i_ix:i_ix+K, j_ix:j_ix+K] += H_vw.T
        full_H[j_ix:j_ix+K, i_ix:i_ix+K] += H_vw
        

  # Add spherical Gaussian prior
  full_H = full_H + np.eye(2 * K * V) * e.lambda0
  return full_H

def fixed_inverse_hessian(H, K):
  elim = K * K
  Sigma = np.linalg.inv(H)
  Sigma_aug = Sigma[:-elim, :-elim]
  Sigma_12 = Sigma[:-elim, -elim:]
  Sigma_22_inv = H[-elim:, -elim:]
  return Sigma_aug - Sigma_12 @ Sigma_22_inv @ Sigma_12.T


def laplace_approx(e, data, samples=None, rotational_fix=True):
  LOGGER.info("Calculate Hessian")
  H = full_hessian(e, data)
  K = e.dimensionality
  V = len([wd for wd in e.vocabulary if "_c" not in wd])
  LOGGER.debug(f"K: {K}, V: {V}")
  Sigma = None
  if rotational_fix:
    LOGGER.train("Freeze last K context vectors")
    LOGGER.info("Invert Hessian...")
    Sigma = fixed_inverse_hessian(H, K)
  else:
    LOGGER.info("Invert Hessian...")
    Sigma = np.linalg.inv(H)

  LOGGER.info("Done!")
  if samples is None:
    return Sigma
  else:
    LOGGER.train(f"Sample {samples} samples")
    inv_vocab = bidict.bidict(e.vocabulary).inv
    L_size = Sigma.shape[0]

    vals, vecs = np.linalg.eigh(Sigma)
    maxval = np.min(vals)
    minval = np.max(vals)
    posvals = np.sum(vals > 0.0)
    negvals = np.sum(vals < 0.0)
    if maxval * minval < 0.00:
      LOGGER.warning(f"Inverse Hessian not positive definite: eigenvalues range [{minval} , {maxval}]")
      LOGGER.warning(f"Positive eigenvals: {np.sum(vals > 0.0)}, negative eigenvals {np.sum(vals < 0.0)}")
      #LOGGER.warning(f"{sorted(vals)}")

    #if np.abs(minval) > np.abs(maxval):
    #  vals = - vals
    if posvals < negvals:
      vals = - vals

    # Replacing negative eigenvalues with zeros gives us
    # the closest positive semidefinite matrix
    vals = np.maximum(vals, np.zeros(vals.shape))
    L = vecs @ np.diag(np.sqrt(vals))

    V_prime = L_size // K
    assert V_prime <= V * 2
    for _ in range(samples):
      e_sample = copy.deepcopy(e)
      deviation = L @ np.random.randn(L_size)
      deviation = deviation.reshape(V_prime, K)
      wordlist = [inv_vocab[ix] for ix in range(V_prime)]

      e_sample[wordlist] = e_sample[wordlist] + deviation
      yield e_sample

def laplace_approx_sigma(e, data, samples=None, rotational_fix=True):
  LOGGER.info("Calculate Hessian")
  H = full_hessian(e, data)
  K = e.dimensionality
  V = len([wd for wd in e.vocabulary if "_c" not in wd])
  LOGGER.debug(f"K: {K}, V: {V}")
  Sigma = None
  if rotational_fix:
    LOGGER.train("Freeze last K context vectors")
    LOGGER.info("Invert Hessian...")
    Sigma = fixed_inverse_hessian(H, K)
  else:
    LOGGER.info("Invert Hessian...")
    Sigma = np.linalg.inv(H)

  LOGGER.info("Done!")
  return Sigma


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
  """
  Find embedding directions which correspond to rotational gradients.
  """
  dim = e.theta.shape[1]
  vs = [_sample_rotation_tangent(e) for _ in range(dim * (dim-1) // 2)]
  vs = np.array(vs)

  q, _ = np.linalg.qr(vs.T)

  return q
