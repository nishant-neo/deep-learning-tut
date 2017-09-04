from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)#Sets the threshold for what messages will be logged

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    #Input Layer
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    
    #Conv layer 1
    conv1 = tf.layers.conv2d(inputs = input_layer, 
                            filters = 32,
                            kernel_size = [5,5],
                            padding = 'same',
                            activation = tf.nn.relu)
    #output tensor produced by conv2d() has a shape of [batch_size, 28, 28, 32]
    
    #Pool layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    #conv layer & pooling
    conv2 = tf.layers.conv2d(
                            inputs = pool1,
                            filters = 64,   #The filters argument specifies the number of filters to apply (here, 32)
                            kernel_size = [5, 5],
                            padding = "same", #we set padding=same here, which instructs TensorFlow to add 0 values to the edges of the output tensor to preserve width and height of 28. 
                            activation = tf.nn.relu) 
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    #Given tensor, this operation returns a tensor that has the same values as tensor with shape.
    #If one component of shape is the special value -1, the size of that dimension is computed so that the total size 
    # remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    # If shape is 1-D or higher, then the operation returns a tensor with shape shape filled with the values of tensor. 
    # In this case, the number of elements implied by shape must be the same as the number of elements in tensor.
    
    #Dense Layer
    #Before we connect the layer, however, we'll flatten our feature map (pool2) to shape [batch_size, features], so that our tensor has only two dimensions.
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout( inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN )
    
    #Logits layer
    logits = tf.layers.dense( inputs = dropout, units = 10)
    
    predictions = {      # Generate predictions (for PREDICT and EVAL mode)
        #We can find the index of max raw element using the tf.argmax function
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor") }
    #Our final output tensor of the CNN, logits, has shape [batch_size, 10].
    
    if mode == tf.estimator.ModeKeys.PREDICT: #Standard names for model modes. The following standard keys are defined: 1. TRAIN: training mode. EVAL: evaluation mode.  PREDICT: inference mode.
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10) #Returns a one-hot tensor.
    loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels, logits=logits ) #takes onehot_labels and logits as arguments, performs softmax activation on logits, calculates cross-entropy, and returns our loss as a scalar Tensor

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
                    
                    
def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    #Estimator offers classes you can instantiate to quickly configure common model types such as regressors and classifiers:
    #The model_fn argument specifies the model function to use for training, evaluation, and prediction; 
    #we pass it the cnn_model_fn we created above.  The model_dir argument specifies the directory where model data (checkpoints) 
    #will be saved (here, we specify the temp directory /tmp/mnist_convnet_model, but feel free to change to another directory of your choice).
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
                                                model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
                                                tensors=tensors_to_log, every_n_iter=50)
         
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                        x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=100,
                                                        num_epochs=None,
                                                        shuffle=True)
    mnist_classifier.train(
                            input_fn=train_input_fn,
                            steps=20000,
                            hooks=[logging_hook])   

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                        x={"x": eval_data},
                                                        y=eval_labels,
                                                        num_epochs=1,
                                                        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
         


if __name__ == "__main__":
    tf.app.run()