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


D = 2
alpha, rho = np.random.randn(D) * 0.1, np.random.randn(D) * 0.1
n_plus, n_minus = 2, 3

hessian = subhessian(n_plus, n_minus, rho, alpha)
print(hessian)

hessian = subhessian(n_plus, n_minus, rho, alpha, j=0)
print(hessian)