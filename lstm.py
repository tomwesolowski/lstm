#
# Simple LSTM without using Tensorflow API
#

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import trange

batch_size = 32
series_size = 16
num_batches = 1000
num_epochs = 4
total_sequence_size = num_batches * batch_size * series_size
state_size = 128
offset = 5
num_classes = 2
plot_width, plot_height = 20, 20


class SmartDict(dict):
    def __init__(self, *args, **kwargs):
        super(SmartDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_data():
    x = np.random.choice(2, total_sequence_size)
    y = np.roll(x, offset)
    y[:offset] = 0

    x = np.reshape(x, (batch_size, -1))
    y = np.reshape(y, (batch_size, -1))
    return x, y


def build_model():
    model = SmartDict()

    model.x_placeholder = tf.placeholder(
        tf.float32, shape=[batch_size, series_size], name="x_placeholder")
    model.y_placeholder = tf.placeholder(
        tf.int32, shape=[batch_size, series_size], name="y_placeholder")

    model.hidden_state = tf.placeholder(
        tf.float32, shape=[batch_size, state_size], name="hidden_state")
    model.cell_state = tf.placeholder(
        tf.float32, shape=[batch_size, state_size], name="cell_state")

    model.inputs_series = tf.unstack(model.x_placeholder, axis=1)
    model.labels_series = tf.unstack(model.y_placeholder, axis=1)

    model.W = tf.get_variable(
        name="W", 
        shape=(1+state_size, 4*state_size),
        initializer=tf.random_uniform_initializer(-0.05, 0.05))
    model.b = tf.get_variable(
        name="b", 
        shape=(1, 4*state_size),
        initializer=tf.constant_initializer(1.0))

    model.current_state = (model.cell_state, model.hidden_state)
    model.states = []

    with tf.variable_scope("lstm_cell"):
      for current_input in model.inputs_series:
        current_input = tf.reshape(current_input, [batch_size, 1])

        c, h = model.current_state 

        forget, input, candidate, output = tf.split(
            tf.matmul(tf.concat([h, current_input], 1), model.W) + model.b, 
            num_or_size_splits=4, axis=1)

        c = tf.nn.sigmoid(forget) * c + \
            tf.nn.sigmoid(input) * tf.tanh(candidate)
        h = tf.tanh(c) * tf.nn.sigmoid(output)

        model.current_state = (c, h)
        model.states.append(h)

    model.W2 = tf.get_variable("W2", shape=(state_size, num_classes))
    model.b2 = tf.get_variable("b2", shape=(1, num_classes))

    model.logits_series = [
        tf.matmul(state, model.W2) + model.b2 for state in model.states]
    model.predictions = [
        tf.nn.softmax(logits) for logits in model.logits_series]

    model.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
        for logits, labels in zip(model.logits_series, model.labels_series)]
    model.total_loss = tf.reduce_mean(model.losses)
    model.opt = tf.train.AdagradOptimizer(1e-1).minimize(model.total_loss)

    model.writer = tf.summary.FileWriter(
        "lstm", tf.get_default_graph(), flush_secs=5)
    model.writer.flush()

    return model


def main():
    # Build model.
    model = build_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        average_loss = 1e3
        saved_predictions = None
        x, y = get_data()

        # Training.
        for epoch in range(num_epochs):
            x, y = get_data()
            cell_state = np.zeros((batch_size, state_size))
            hidden_state = np.zeros((batch_size, state_size))

            tr = trange(num_batches)
            tr.set_description("Epoch %d/%d" % (epoch + 1, num_epochs))

            for batch_id in tr:
                batch_x = x[:, batch_id * series_size:(batch_id + 1) * series_size]
                batch_y = y[:, batch_id * series_size:(batch_id + 1) * series_size]

                _, loss, predictions, current_state = sess.run(
                    [model.opt, model.total_loss,
                     model.predictions, model.current_state],
                    feed_dict={
                        model.x_placeholder: batch_x,
                        model.y_placeholder: batch_y,
                        model.cell_state: cell_state,
                        model.hidden_state: hidden_state,
                    })
                cell_state, hidden_state = current_state

                average_loss = 0.99 * average_loss + 0.01 * loss

                if batch_id % 10 == 0:
                    tr.set_postfix(loss=average_loss)

                if epoch + 1 == num_epochs:
                    if saved_predictions is None:
                        saved_predictions = predictions
                    else:
                        saved_predictions = np.concatenate(
                            (saved_predictions, predictions), axis=0)

        # Plot results.
        plot_batch = 0
        plot_x = x[plot_batch, :-offset]
        plot_predictions = np.argmax(saved_predictions[offset:, plot_batch, :],
                                     axis=1)

        plot_x = plot_x[-plot_width * plot_height:]
        plot_predictions = plot_predictions[-plot_width * plot_height:]

        plot_x = np.reshape(plot_x, (plot_height, plot_width))
        plot_predictions = np.reshape(
            plot_predictions, (plot_height, plot_width))

        plot_diff = (plot_x != plot_predictions)
        plots = [plot_x, plot_predictions, plot_diff]
        plots_titles = ['labels', 'predictions', 'diff']

        fig, axes = plt.subplots(1, len(plots))
        for ax, plot, title in zip(axes, plots, plots_titles):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_title(title)
            ax.matshow(plot, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
