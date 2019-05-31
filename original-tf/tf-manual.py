import tensorflow as tf

# 创建变量W, b
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)

# 创建x节点，用来输入x_train[n]
x = tf.placeholder(tf.float32)

# 这个就是模型函数
linear_model = W * x + b

# 创建y节点，用来输入y_train[n]
y = tf.placeholder(tf.float32)

# 创建损失函数，用于评估模型输出值与期望值差距
loss = tf.reduce_sum(tf.square(linear_model - y))

init = tf.global_variables_initializer()

x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

with tf.Session() as sess:
    sess.run(init)

    print('===================round # 1===================')
    r_W = sess.run(W)
    print('W:', r_W)
    r_b = sess.run(b)
    print('b:', r_b)

    r_linear_model = sess.run(linear_model, {x: x_train})
    print('linear_model:', r_linear_model)

    r_loss = sess.run(loss, {x: x_train, y: y_train})
    print('loss:', r_loss)

    print('===================round # 2===================')
    fixW = tf.assign(W, [2.])
    print(fixW)
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])

    r_fixW = sess.run(W)
    print('fix_W:', r_fixW)
    r_fixb = sess.run(b)
    print('fix_b:', r_fixb)

    r_linear_model = sess.run(linear_model, {x: x_train})
    print('linear_model:', r_linear_model)

    r_loss = sess.run(loss, {x: x_train, y: y_train})
    print('loss:', r_loss)
