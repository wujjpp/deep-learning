import tensorflow as tf

# dtype可不填，自动识别
t0 = tf.constant(3, dtype=tf.int32)
t1 = tf.constant([1., 2.3, 3.4], dtype=tf.float32)
t2 = tf.constant([['a', 'b'], ['c', 'd']], dtype=tf.string)
t3 = tf.constant([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])

print(t0)
print(t1)
print(t2)
print(t3)

with tf.Session() as sess:
    print(sess.run(t0))
    print(sess.run(t1))
    print(sess.run(t2))
    print(sess.run(t3))