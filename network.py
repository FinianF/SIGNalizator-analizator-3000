import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 40, 40, 3])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * 64])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=9)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Create the Estimator
sign_classifier = tf.estimator.Estimator(
	model_fn=cnn_model_fn, model_dir="./network-final")


def format_data(images):
	data, labels = zip(*images)
	data = np.array(data, dtype=np.float32)
	labels = list(labels)

	for i, label in enumerate(labels):
		try: labels[i] = label.index(1)
		except: labels[i] = 8

	labels = np.asarray(labels, dtype=np.int32)

	return data, labels

def cnn_train(images):
	data, labels = format_data(images)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": data},
		y=labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	sign_classifier.train(
		input_fn=train_input_fn,
		steps=20000,
		hooks=[logging_hook])


def cnn_eval(images):
	data, labels = format_data(images)

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": data},
		y=labels,
		num_epochs=1,
		shuffle=False)
	eval_results = sign_classifier.evaluate(input_fn=eval_input_fn)

	return eval_results

def cnn_predict(image):
	image = np.array(image, dtype=np.float32)

	input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': image},
		num_epochs=1,
		shuffle=False)
	result = sign_classifier.predict(input_fn=input_fn)

	return result
