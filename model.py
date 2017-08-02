import tensorflow as tf

# Define the size of a square input image
box_size = 32
# Store the total number of classes our classifier needs to learn
num_classes = 10

def load_data(data_abs_path):

    pass

def next_batch(data):

    pass

def train(data_abs_path):

    data = load_data(data_abs_path)

    # Tensorflow placeholders will contain external inputs
    # Placeholder for input image features
    x = tf.placeholder(tf.float32, [None, box_size * box_size])
    # Placeholder for input image's one-hot encoded label
    y = tf.placeholder(tf.float32, [None, num_classes])

    # Neural network weights and biases
    W = tf.Variable(tf.zeros([box_size * box_size, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Hyperparameters
    learning_rate_alpha = 0.01
    numb_epoches = 500

    # Define a learning model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)

    # cross-entropy function will calculate loss value
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

    # Initialize optimizer that will lower loss value and improve performance of our model
    optimizer = tf.train.GradientDescentOptimizer(learning_rate_alpha).minimize(cross_entropy)


    # Compare predicted label (class) with input image's label
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate mean of ...
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Run computational graph
    sess = tf.Session()

    # To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train model
    for epoch in range(numb_epoches):

        batch_xs, batch_ys = next_batch(100)

        # Optimize model's weights and biases
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        # Print accuracy - model's performance on test set
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
