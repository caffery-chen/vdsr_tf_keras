import tensorflow as tf
class NN_Layer:
    def __init__(self):
        pass

    @staticmethod
    def cus_conv2d_layer(input, channels_in, channels_out, kernel_size, name="customized_conv2d"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([kernel_size[0], kernel_size[1], channels_in, channels_out]), name = "W")
            b = tf.Variable(tf.random_normal([channels_out]), name = "B")
            conv = tf.nn.conv2d(input, w, strides = [1,1,1,1], padding = "SAME")
            act = tf.nn.relu(conv + b)
        return act

    @staticmethod
    def cus_fc_layer(input, channels_in, channels_out, name = "customized_fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([channels_in, channels_out]), name = 'W')
            b = tf.Variable(tf.random_normal([channels_out]), name = 'B')
            act = tf.nn.relu(tf.matmul(input, w) + b)
        return act

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape = [None, 784], name = "x")
    x_image = tf.reshape(x, [-1, 28,28,1])
    y = tf.placeholder(tf.float32, shape = [None, 10], name = "labels")

    conv1 = NN_Layer.cus_conv2d_layer(x_image, 1, 32, "conv_layer_1")
    conv2 = NN_Layer.cus_conv2d_layer(conv1, 32, 64, "conv_layer_2")

    flattend = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc1 = NN_Layer.cus_fc_layer(flattend, 7 * 7 * 64, 1024, "fc1")
    logits = NN_Layer.cus_fc_layer(fc1, 1024, 10, 'fc2')

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('cross_entropy', xent)
        tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        batch = mnist.train.next_batch(100)

        if i % 500 ==0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y:batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_true: batch[1]})