#import tensorflow as tf
import numpy as np
import tensorflow as tf

def inv_gd(P, i, lr=0.001, iterations=1000):
    assert P.shape[0] == P.shape[1]
    n = P.shape[0]

    e = np.zeros(n)
    e[i] = 1

    x = (np.random.rand(n) - 0.5) / n

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    current_lr = 0.0
    for iteration in range(iterations):
        current_lr = lr * iteration / iterations
        gradient = (x @ P) - e
        step = optimizer.update_step(gradient)
        print(step)

        x = x - current_lr * gradient / n
    return x

if __name__ == '__main__':
    n = 30
    L = np.random.rand(n,n) * 2
    P = L.T @ L
    Sigma = np.linalg.inv(P)

    i = n // 2
    x = inv_gd(P, i)
    print(x)
    print(Sigma[i])
    #print(P)
    
    print(np.round(P @ x, 2))

