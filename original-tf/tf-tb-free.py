import tensorflow as tf

# 创建x节点，用来输入x_train[n]
x = tf.placeholder(tf.float32, name='x')
# 创建y节点，用来输入y_train[n]
y = tf.placeholder(tf.float32, name='y')

# # 创建变量W, b
W = tf.Variable([.1], dtype=tf.float32, name='W')
b = tf.Variable([-.1], dtype=tf.float32, name='b')

with tf.name_scope('kernel'):
    linear_model = W * x + b

# # 初始化变量
init = tf.global_variables_initializer()

# 训练5000次
epochs = 2
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('results', sess.graph)
    writer.close()
