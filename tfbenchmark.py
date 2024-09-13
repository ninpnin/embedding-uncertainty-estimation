import tensorflow as tf

A = tf.random.normal(shape=(100,100))
print(A)

rounds = 1000
import time
start = time.process_time()
# your code here    

@tf.function
def invsum(A):
    B = tf.linalg.inv(A)
    for r in range(rounds):
        B += tf.linalg.inv(A)
    return B

end = time.process_time() - start
invsum(A)
print(end - start)
print((end - start) / rounds)

end = time.process_time() - start
invsum(A)
print(end - start)
print((end - start) / rounds)