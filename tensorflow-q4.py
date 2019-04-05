import tensorflow as tf
import numpy as np


class NeuralNet:

    def __init__(self, config):
        self.input_units = config.input_units
        self.hidden_units = config.hidden_units
        self.output_units = config.output_units
        self.learning_rate = config.learning_rate

        self.X = tf.placeholder("float64", (None, self.input_units), "features")
        self.Y = tf.placeholder("float64", (None, self.output_units), "labels")

        with tf.variable_scope("forward_propagation"):
            self.weights_i_h = tf.Variable(tf.random_normal([self.input_units, self.hidden_units], dtype="float64"),
                                           name="input-hidden_weights")
            self.biases_h = tf.Variable(tf.random_normal([self.hidden_units], dtype="float64"), name="hidden_bias")
            self.weights_h_o = tf.Variable(tf.random_normal([self.hidden_units, self.output_units], dtype="float64"),
                                           name="hidden-output_weights")
            self.biases_o = tf.Variable(tf.random_normal([self.output_units], dtype="float64"), name="output_bias")

            self.hidden_val = tf.sigmoid(tf.add(tf.matmul(self.X, self.weights_i_h), self.biases_h),
                                         name="activated_hidden_value")
            self.output_val = tf.sigmoid(tf.add(tf.matmul(self.hidden_val, self.weights_h_o), self.biases_o),
                                         name="activated_output_value")

        with tf.variable_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.Y, self.output_val)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="GradientDescent").minimize(
                self.loss)

    def predict(self, x, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output_val, {self.X: [x]})

    def update(self, x, y, sess=None):
        sess = sess or tf.get_default_session()
        optimizer, loss, output = sess.run([self.optimizer, self.loss, self.output_val],
                                           feed_dict={self.X: [x], self.Y: [y]})
        return loss, output


def read_csv(filename):
    with open(filename) as file:
        data = [line[:-1].split(',') for line in file.readlines()]
        data = np.array(data).astype('float64')
    return data


def make_output_data(data):
    val = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0]}
    label = [val[i] for i in data[:, 0].astype('int')]
    return np.array(label)


def train(train_x, train_y, test_x, test_y, l_rate, epoch, hidden_units):
    flags = tf.flags

    flags.DEFINE_integer("input_units", train_x.shape[1], "size of input layer")
    flags.DEFINE_integer("hidden_units", hidden_units, "size of hidden layer")
    flags.DEFINE_integer("output_units", train_y.shape[1], "size of output layer")
    flags.DEFINE_float("learning_rate", l_rate, "Learning rate")
    config = flags.FLAGS

    network = NeuralNet(config)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for i in range(epoch):
            accuracy = []

            for x, y in zip(train_x, train_y):
                loss, output = network.update(x, y, sess)
                accuracy.append(np.argmax(output) == np.argmax(y))
            print("Training accuracy: ", np.mean(accuracy))
            print("Testing accuracy: ", evaluate(network, test_x, test_y))
            print()

    return network


def evaluate(model, x, y, sess=None):
    predictions = []
    for i, j in zip(x, y):
        output = model.predict(i, sess)
        predictions.append(j == np.argmax(output)+1)
    return np.mean(predictions)


wine_train = read_csv("../train_wine.csv")
wine_test = read_csv("../test_wine.csv")

train_output = make_output_data(wine_train)
wine_train = wine_train[:, 1:]

nmax = np.max(np.vstack((wine_train, wine_test[:, 1:])), axis=0)
nmin = np.min(np.vstack((wine_train, wine_test[:, 1:])), axis=0)

normalized_wine = (wine_train - nmin) / (nmax - nmin)
norm_wine_test = (wine_test[:, 1:] - nmin) / (nmax - nmin)

net = train(normalized_wine, train_output, norm_wine_test, wine_test[:, 0], 0.1, 50, 4)

