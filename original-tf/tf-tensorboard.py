import tensorflow as tf
from progress.bar import Bar

# 创建变量W, b
W = tf.Variable([.1], dtype=tf.float32, name='W')
b = tf.Variable([-.1], dtype=tf.float32, name='b')

# 创建x节点，用来输入x_train[n]
x = tf.placeholder(tf.float32, name='x')

# 这个就是模型函数
linear_model = W * x + b

# 创建y节点，用来输入y_train[n]
y = tf.placeholder(tf.float32, name='y')

with tf.name_scope('loss-model'):
    # 创建损失函数，用于评估模型输出值与期望值差距
    loss = tf.reduce_sum(tf.square(linear_model - y))
    tf.summary.scalar('loss', loss)

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)

train = optimizer.minimize(loss)

# 训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

# 初始化变量
init = tf.global_variables_initializer()

# 训练5000次
epochs = 2
with tf.Session() as sess:
    sess.run(init)

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('results', sess.graph)

    with Bar('Processing', max=epochs) as bar:
        for i in range(epochs):

            summary, _ = sess.run([merged, train], {x: x_train, y: y_train})

            writer.add_summary(summary, i)

            bar.next()

    r_W, r_b, r_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print('W:', r_W)
    print('b:', r_b)
    print('loss:', r_loss)
