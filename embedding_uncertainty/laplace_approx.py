import tensorflow as tf
import numpy as np

def subhessian(n_plus, n_minus, rho, alpha, i=0, j=1):
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
