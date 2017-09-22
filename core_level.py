import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cur_dir = os.path.dirname(os.path.abspath(__file__))
sess = tf.Session()

# model
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# real result
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)

# loss
loss = tf.reduce_sum(squared_deltas)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# initialize
init = tf.global_variables_initializer()
sess.run(init)

# train
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# run
print(sess.run([W, b]))

# graph
train_writer = tf.summary.FileWriter(cur_dir + '/train', sess.graph)
