import tensorflow as tf
import numpy as np
D, V = 2, 5

np.random.seed(0)

rho = np.random.randn(V, D)
alpha = np.random.randn(V, D)
W = tf.Variable(np.random.randn(D,D))

def loss0(r):
    return tf.reduce_sum(tf.multiply(r, r))

with tf.GradientTape() as g:
    l = loss0(rho @ W)
    dl_dW = g.gradient(l, W)

def gradient0(rho, W):
    return tf.transpose(2 * tf.transpose(W) @ rho.T @ rho)

def gradient1(alpha, W):
    Winv = tf.linalg.inv(W)
    WinvT = tf.transpose(Winv)
    return - 2 * WinvT @ Winv @ alpha.T @ alpha @ WinvT

print(l)
print(dl_dW)
print(tf.linalg.trace(tf.transpose(W) @ rho.T @ rho @ W))
print(gradient0(rho, W))

with tf.GradientTape() as g:
    l = loss0(alpha @ tf.linalg.inv(W))
    dl_dW = g.gradient(l, W)

print(l)
print(dl_dW)
print(tf.linalg.trace(tf.transpose(tf.linalg.inv(W)) @ alpha.T @ alpha @ tf.linalg.inv(W)))
print(gradient1(alpha, W))


