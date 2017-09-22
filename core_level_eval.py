import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cur_dir = os.path.dirname(os.path.abspath(__file__))
sess = tf.Session()

def evaluate(x, y, w, b):
    if len(x) != len(y):
        return None

    sum_of_loss = 0.
    for i in range(len(x)):
        res = w * x[i] + b
        sum_of_loss += (y[i] - res) * (y[i] - res)

    return sum_of_loss / len(x)

# model
W = tf.Variable([.3], dtype=tf.float64)
b = tf.Variable([-.3], dtype=tf.float64)
x = tf.placeholder(tf.float64)
linear_model = W * x + b

# real result
y = tf.placeholder(tf.float64)
squared_deltas = tf.square(linear_model - y)

# loss
loss = tf.reduce_sum(squared_deltas)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# initialize
init = tf.global_variables_initializer()
sess.run(init)

# data set
x_train = [1., 2., 3., 4.]
y_train = [0., -1., -2., -3.]
x_eval = [2., 5., 8., 1.]
y_eval = [-1.01, -4.1, -7, 0.]

# train
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# result
(w_ret, b_ret) = sess.run([W, b])

# evaluation
loss_1 = evaluate(x_train, y_train, w_ret,  b_ret)
loss_2 = evaluate(x_eval, y_eval, w_ret,  b_ret)

print(loss_1[0])
print(loss_2[0])

# graph
train_writer = tf.summary.FileWriter(cur_dir + '/train', sess.graph)
