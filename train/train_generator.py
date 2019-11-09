import tensorflow as tf
import numpy as np

from train.Model import SongGenerator
from train.PrepData import PrepData
from constants import EPOCHS
from MidiNumpyConversion.NumpyToMidi import NumpyToMidi
from Visualization.SongDisplay import SongDisplay

# Load data
# data_manager = PrepData("data/npy/", 2294)
data_manager = PrepData("data/npy/", 2)
data_manager.load_data()

generator = SongGenerator()
loss_obj = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
# TODO: more tf.keras.metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")
test_loss = tf.keras.metrics.Mean(name="test_loss")
train_recall = tf.keras.metrics.Recall(name="train_recall")
test_recall = tf.keras.metrics.Recall(name="test_recall")
train_precision = tf.keras.metrics.Precision(name="train_precision")
test_precision = tf.keras.metrics.Precision(name="test_precision")


@tf.function
def train_step(pc: tf.Tensor, song: tf.Tensor):
    """
    Performs one training step with the `pc` input and `song` label.
    """
    with tf.GradientTape() as tape:
        generated_song = generator(pc)
        loss = loss_obj(song, generated_song)
    gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    train_loss(loss)
    train_recall(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))
    train_precision(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))


@tf.function
def test_step(pc: tf.Tensor, song: tf.Tensor):
    """
    Performs one test step with the `pc` input and `song` label.
    """
    generated_song = generator(pc)
    t_loss = loss_obj(song, generated_song)
    test_loss(t_loss)
    test_recall(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))
    test_precision(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))
    return generated_song


for epoch in range(EPOCHS):
    for pc, song in data_manager.train_ds:
        train_step(pc, song)

    for pc, song in data_manager.test_ds:
        generated_song = test_step(pc, song).numpy()
        np.save("output/test_epoch_" + str(epoch + 1), generated_song)
        if epoch == EPOCHS - 1:
            SongDisplay.show(song.numpy().reshape((16, 96, 96)).astype('int8'))
            SongDisplay.show(generated_song.reshape((16, 96, 96)).astype('int8'))

    # TODO: actual logs
    print("Epoch: ", epoch + 1,
          "Train loss:", train_loss.result().numpy(),
        #   "Test loss:", test_loss.result().numpy(),
          "Train recall:", train_recall.result().numpy(),
        #   "Test recall:", test_recall.result().numpy(),
          "Train prec:", train_precision.result().numpy(),
        #   "Test prec:", test_precision.result().numpy()
          )

    train_loss.reset_states()
    test_loss.reset_states()
    train_recall.reset_states()
    test_recall.reset_states()
    train_precision.reset_states()
    test_precision.reset_states()

ntm = NumpyToMidi()
song = np.load("output/test_epoch_10.npy").reshape((16, 96, 96)).astype('int8')
midi = ntm.numpy_to_midi(song)
SongDisplay.show(song)
midi.save("data/temp/bak.mid")

print("Done!")
