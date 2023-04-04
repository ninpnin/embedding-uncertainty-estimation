import tensorflow as tf

def calculate_hessian(x=None, l=None):
    if x is None:
        x = tf.random.normal([2])
    print(x)
    x = tf.Variable(x)

    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        loss = None
        if l is None:
            r = tf.linalg.norm(x)
            loss = tf.exp(- ((r - 2) ** 2 ))
        else:
            loss = l(x)

      g = t1.gradient(loss, x)

    h = t2.jacobian(g, x)

    print(h)

x_0 = tf.constant([2.0, 0.0])
calculate_hessian()
calculate_hessian(x_0)
calculate_hessian(x_0, l=lambda x: tf.sigmoid(-tf.reduce_sum(tf.multiply(x, x))))