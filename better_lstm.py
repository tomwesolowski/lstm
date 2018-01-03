#
# Simple LSTM without using Tensorflow API
# with peephole connections and cell clipping.
#

import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import trange

batch_size = 32
series_size = 16
num_batches = 1000
num_epochs = 3
total_sequence_size = num_batches * batch_size * series_size
state_size = 128
offset = 5
num_classes = 4
plot_width, plot_height = 20, 20
use_peephole = True
cell_clip = 5.0 # disabled if None 
flush_stats_every = 10
save_stats = True

class SmartDict(dict):
    def __init__(self, *args, **kwargs):
        super(SmartDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_date():
    return datetime.datetime.now().strftime('%02m-%02d_%02H-%02M-%02S')

def get_data():
    x = np.random.choice(num_classes, total_sequence_size)
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

    if use_peephole:
      model.f_peep = tf.get_variable(
        name="f_peep", 
        shape=[state_size],
        dtype=tf.float32)

      model.i_peep = tf.get_variable(
        name="i_peep", 
        shape=[state_size],
        dtype=tf.float32)

      model.o_peep = tf.get_variable(
        name="o_peep", 
        shape=[state_size],
        dtype=tf.float32)

    model.current_state = (model.cell_state, model.hidden_state)
    model.states = []

    with tf.variable_scope("lstm_cell"):
      for roll_step, current_input in enumerate(model.inputs_series):
        current_input = tf.reshape(current_input, [batch_size, 1])

        c, h = model.current_state 

        forget, input, candidate, output = tf.split(
            tf.matmul(tf.concat([h, current_input], 1), model.W) + model.b, 
            num_or_size_splits=4, axis=1)

        if use_peephole:
          new_c = tf.nn.sigmoid(forget + model.f_peep * c) * c + \
              tf.nn.sigmoid(input + model.i_peep * c) * tf.tanh(candidate)
        else:
          new_c = tf.nn.sigmoid(forget) * c + \
              tf.nn.sigmoid(input) * tf.tanh(candidate)

        if cell_clip:
          new_c = tf.clip_by_value(new_c, -cell_clip, cell_clip)

        new_h = tf.tanh(new_c) * tf.nn.sigmoid(output + model.o_peep * c)

        tf.summary.histogram('h%d' % roll_step, tf.reshape(h, [-1]))
        tf.summary.histogram('c%d' % roll_step, tf.reshape(c, [-1]))

        model.current_state = (new_c, new_h)
        model.states.append(new_h)

    model.Wproj = tf.get_variable("Wproj", shape=(state_size, num_classes))
    model.bproj = tf.get_variable("bproj", shape=(1, num_classes))

    model.logits_series = [
        tf.matmul(state, model.Wproj) + model.bproj for state in model.states]
    model.predictions = [
        tf.nn.softmax(logits) for logits in model.logits_series]

    model.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
        for logits, labels in zip(model.logits_series, model.labels_series)]
    model.total_loss = tf.reduce_mean(model.losses)
    model.opt = tf.train.AdagradOptimizer(1e-1).minimize(model.total_loss)

    model.merged_summaries = tf.summary.merge_all()
    model.writer = tf.summary.FileWriter(
        "better_lstm_%s" % get_date(), tf.get_default_graph(), flush_secs=5)
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

                _, loss, predictions, current_state, summaries = sess.run(
                    [model.opt, model.total_loss,
                     model.predictions, model.current_state,
                     model.merged_summaries],
                    feed_dict={
                        model.x_placeholder: batch_x,
                        model.y_placeholder: batch_y,
                        model.cell_state: cell_state,
                        model.hidden_state: hidden_state,
                    })
                cell_state, hidden_state = current_state

                average_loss = 0.99 * average_loss + 0.01 * loss

                if batch_id % flush_stats_every == 0:
                    tr.set_postfix(loss=average_loss)
                    if save_stats:
                      model.writer.add_summary(
                        summaries, 
                        (epoch * num_epochs + batch_id) // flush_stats_every)

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
