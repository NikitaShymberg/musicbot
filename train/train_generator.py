import tensorflow as tf

from train.Model import SongGenerator
from train.PrepData import PrepData
from constants import EPOCHS

# Load data
# data_manager = PrepData("data/npy/", 22229)
data_manager = PrepData("data/npy/", 750)
data_manager.load_data()

generator = SongGenerator()
loss_obj = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
# TODO: more tf.keras.metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")
test_loss = tf.keras.metrics.Mean(name="test_loss")


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


@tf.function
def test_step(pc: tf.Tensor, song: tf.Tensor):
    """
    Performs one test step with the `pc` input and `song` label.
    """
    generated_song = generator(pc)
    t_loss = loss_obj(song, generated_song)
    test_loss(t_loss)


for epoch in range(EPOCHS):
    for pc, song in data_manager.train_ds:
        train_step(pc, song)

    for pc, song in data_manager.test_ds:
        train_step(pc, song)

    # TODO: actual logs
    print("Epoch: ", epoch + 1,
          "Train loss:", train_loss.result(),
          "Test loss:", test_loss.result())

    train_loss.reset_states()
    test_loss.reset_states()

print("Done!")
