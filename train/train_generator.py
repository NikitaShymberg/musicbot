"""
This is the main script that trains the song generator network.
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from train.Model import SongGenerator
from train.PrepData import PrepData
from constants import EPOCHS
from MidiNumpyConversion.NumpyToMidi import NumpyToMidi
from Visualization.SongDisplay import SongDisplay

# Load data
data_manager = PrepData("data/npy/", 2294)
# data_manager = PrepData("data/npy/", 2)
data_manager.load_data()

# Set up all training variables in a global scope
generator = SongGenerator()
loss_obj = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

# Metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")
test_loss = tf.keras.metrics.Mean(name="test_loss")
train_recall = tf.keras.metrics.Recall(name="train_recall")
test_recall = tf.keras.metrics.Recall(name="test_recall")
train_precision = tf.keras.metrics.Precision(name="train_precision")
test_precision = tf.keras.metrics.Precision(name="test_precision")

# Tensorboard
train_writer = tf.summary.create_file_writer("./logs/training")
test_writer = tf.summary.create_file_writer("./logs/testing")


@tf.function
def train_step(pc: tf.Tensor, song: tf.Tensor):
    """
    Performs one training step with the `pc` input and `song` label.
    Also updates train metrics.
    """
    with tf.GradientTape() as tape:
        generated_song = generator(pc)
        loss = loss_obj(song, generated_song)
    gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    # Metrics
    train_loss(loss)
    train_recall(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))
    train_precision(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))


@tf.function
def test_step(pc: tf.Tensor, song: tf.Tensor):
    """
    Performs one test step with the `pc` input and `song` label.
    Also updates test metrics.
    Returns the generated test song.
    """
    generated_song = generator(pc)
    t_loss = loss_obj(song, generated_song)
    # Metrics
    test_loss(t_loss)
    test_recall(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))
    test_precision(tf.cast(song, tf.bool), tf.cast(generated_song, tf.bool))

    return generated_song


def train_logs(epoch):
    """
    Records all tensorboard logs for training metrics at the given `epoch`.
    """
    with train_writer.as_default():
        tf.summary.scalar("training_loss", train_loss.result(), step=epoch)
        print("Training loss:", train_loss.result().numpy())
        tf.summary.scalar("training_recall - low means many missed notes",
                          train_recall.result(), step=epoch)
        tf.summary.scalar("training_precision - low means many extra notes",
                          train_precision.result(), step=epoch)
    train_loss.reset_states()
    train_recall.reset_states()
    train_precision.reset_states()


def test_logs(epoch):
    """
    Records all tensorboard logs for testing metrics at the given `epoch`.
    """
    with test_writer.as_default():
        tf.summary.scalar("testing_loss", test_loss.result(), step=epoch)
        tf.summary.scalar("testing_recall - low means many missed notes",
                          test_recall.result(), step=epoch)
        tf.summary.scalar("testing_precision - low means many extra notes",
                          test_precision.result(), step=epoch)
    test_loss.reset_states()
    test_recall.reset_states()
    test_precision.reset_states()


print("Starting training...")
for epoch in tqdm(range(EPOCHS)):
    for pc, song in data_manager.train_ds:
        train_step(pc, song)
    train_logs(epoch + 1)

    displayed = False
    for pc, song in data_manager.test_ds:
        generated_song = test_step(pc, song).numpy()
        # np.save("output/test_epoch_" + str(epoch + 1), generated_song)
        if epoch == EPOCHS - 1 and not displayed:
            displayed = True
            SongDisplay.show(PrepData.to_song(song[0].numpy()))
            SongDisplay.show(PrepData.to_song(generated_song[0]))
    test_logs(epoch + 1)


print("Done!")
