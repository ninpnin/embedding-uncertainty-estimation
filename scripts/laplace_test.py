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
alpha = np.random.randn(D) / np.sqrt(D)
rho = np.random.randn(D) / np.sqrt(D)

print("rho", alpha)
print("alpha", rho)

p = tf.sigmoid(tf.reduce_sum(tf.multiply(rho, alpha)))
print("p", p)


n_plus = int(p * 100)
n_minus = int((1-p) * 100)


hessian_ij = subhessian(n_plus, n_minus, rho, alpha)
print(hessian_ij)

hessian_ii = subhessian(n_plus, n_minus, rho, alpha, j=0)
print(hessian_ii)

hessian_jj = subhessian(n_plus, n_minus, rho, alpha, i=1)
print(hessian_jj)

precision = np.zeros([D*2 , D*2])
precision[:D, :D] = hessian_ii
precision[D:, D:] = hessian_jj
precision[:D, D:] = hessian_ij
precision[D:, :D] = hessian_ij.numpy().T

print(precision)


covariance = np.linalg.inv(precision)

print(covariance)