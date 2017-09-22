import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cur_dir = os.path.dirname(os.path.abspath(__file__))
sess = tf.Session()

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("i", shape=[1, 2])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir=cur_dir + '/model')

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
i_train = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
o_train = np.array([0., 1., 1., 0.])
i_eval = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
o_eval = np.array([0., 1., 1., 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"i": i_train}, o_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"i": i_train}, o_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"i": i_eval}, o_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
#train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

#print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

# graph
train_writer = tf.summary.FileWriter(cur_dir + '/train', sess.graph)