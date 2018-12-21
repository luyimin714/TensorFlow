import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

c = tf.concat([a, b], -1)

d = tf.constant([[1, 2, 3],
                 [4, 5, 6]])

e = tf.reshape(d, shape=[-1, 2])

f = tf.constant([[7,  8,  9],
                 [10, 11, 12]])

g = tf.concat([d, f], -1)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(e))
    print(sess.run(g))

print(tf.__version__)