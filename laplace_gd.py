import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import progressbar

DIM = 2
VOCAB_SIZE = 2
LAMBDA0 = 0.1

def posterior(x, lambda0=1.0):
    nu = tf.reduce_sum(tf.multiply(x[:DIM], x[DIM:]))
    ll = tf.math.log(tf.math.sigmoid(nu))
    prior = - lambda0 * tf.linalg.norm(x) ** 2.0
    return ll + prior

def gradient_descent(iterations=100, lambda0=1.0):
    x = np.random.randn(VOCAB_SIZE * DIM)
    x = tf.Variable(x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    current_lr = 0.0
    for iteration in progressbar.progressbar(range(iterations)):
        loss = lambda: - posterior(x, lambda0)
        # Call minimize to update the list of variables.
        optimizer.minimize(loss, var_list=[x])
        #print(x)

    return x

def calculate_hessian(x, l=None):
    x = tf.Variable(x)
    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        loss = l(x)

      g = t1.gradient(loss, x)
    return t2.jacobian(g, x)

x_0 = gradient_descent(lambda0=LAMBDA0, iterations=5000)
print(x_0)
hess = calculate_hessian(x_0, l= lambda theta: posterior(theta, lambda0=LAMBDA0))
print(hess)

values, vectors = tf.linalg.eig(hess)
values, vectors = tf.math.real(values), tf.math.real(vectors)
print(values)
print(vectors)

precision = 4
values = np.round(values.numpy(), precision)
vectors = np.round(vectors.numpy(), precision)

print(values)
print(vectors)

# The Hessian of a linear transformation A: X -> Y is A^T H(x) A

indices = [i for i in range(VOCAB_SIZE * DIM) if values[i] != 0]
print(indices)
A = tf.gather(tf.transpose(vectors), indices)
print(A)