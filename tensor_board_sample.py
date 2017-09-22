import os
import tensorflow as tf

cur_dir = os.path.dirname(os.path.abspath(__file__))
sess = tf.Session()

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

print(sess.run(node3))

# graph
train_writer = tf.summary.FileWriter(cur_dir + '/train', sess.graph)
