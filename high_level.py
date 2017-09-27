import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cur_dir = os.path.dirname(os.path.abspath(__file__))
sess = tf.Session()

# feature list 정의
feature_columns = [tf.feature_column.numeric_column("x")]

# Estimator 생성
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir=cur_dir+'/train')

# data set 준비
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# Training 을 위한 input function 정의
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# training
estimator.train(input_fn=input_fn, steps=1000)

# 결과 평가
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

# prediction
x_predict = np.array([10., -5.])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_predict}, batch_size=4, num_epochs=1, shuffle=False)
predictions = estimator.predict(input_fn=predict_input_fn)

for p in predictions:
    print("{}".format(p['predictions']))

# graph
train_writer = tf.summary.FileWriter(cur_dir + '/train', sess.graph)
