import tensorflow as tf

node1 = tf.constant(1.2)
node2 = tf.constant(3.4)

adder = node1 + node2
print(adder)

with tf.Session() as sess:
    print(sess.run(adder))
