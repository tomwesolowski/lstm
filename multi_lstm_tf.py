#
# Simple LSTM with use of Tensorflow API.
#

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import trange

batch_size = 32
series_size = 16
num_batches = 1000
num_epochs = 3
total_sequence_size = num_batches * batch_size * series_size
state_size = 8
offset = 5
num_classes = 4
num_layers = 3
plot_width, plot_height = 20, 20


class SmartDict(dict):
  def __init__(self, *args, **kwargs):
    super(SmartDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


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
      tf.float32, shape=[batch_size, series_size])
  model.y_placeholder = tf.placeholder(
      tf.int32, shape=[batch_size, series_size])

  model.init_state = tf.placeholder(
      tf.float32, shape=[num_layers, 2, batch_size, state_size])

  model.inputs_series = tf.split(model.x_placeholder, series_size, axis=1)
  model.labels_series = tf.unstack(model.y_placeholder, axis=1)

  model.layers_states = tf.unstack(model.init_state, axis=0)
  model.tuples_states = tuple(
      [tf.nn.rnn_cell.LSTMStateTuple(state[0], state[1]) 
      for state in model.layers_states]
  )

  model.cell = tf.nn.rnn_cell.MultiRNNCell(
      [tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)
  model.states, model.current_state = tf.nn.static_rnn(
      model.cell, model.inputs_series, initial_state=model.tuples_states)

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
  model.opt = tf.train.AdagradOptimizer(0.1).minimize(model.total_loss)

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
      current_state = np.zeros((num_layers, 2, batch_size, state_size))

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
                model.init_state: current_state
            })

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
