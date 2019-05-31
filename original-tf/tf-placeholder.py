import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = a + b

triple = add * 3

with tf.Session() as sess:
    r1 = sess.run(add, {a: 3, b: 4})
    print(r1)

    r2 = sess.run(add, {a: [1, 2], b: [3, 4]})
    print(r2)

    r3 = sess.run(triple, {a: 3, b: 4})
    print(r3)

    r4 = sess.run(triple, {a: [1, 2], b: [3, 4]})
    print(r4)
