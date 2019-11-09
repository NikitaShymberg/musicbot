"""
This class contains the music generation model
"""
import tensorflow as tf
from constants import FLAT_SIZE


class SongGenerator(tf.keras.Model):
    """
    This class defines the model structure
    """
    def __init__(self):
        """
        Initializes the model object.
        TODO: mess around with this.
        """
        super(SongGenerator, self).__init__()
        self.d1 = tf.keras.layers.Dense(FLAT_SIZE // 10000, activation='relu')
        self.d2 = tf.keras.layers.Dense(FLAT_SIZE // 10000, activation='relu')
        self.d3 = tf.keras.layers.Dense(FLAT_SIZE // 10000, activation='relu')
        self.d4 = tf.keras.layers.Dense(FLAT_SIZE, activation='sigmoid')

    def call(self, x):
        """
        Runs input `x` through the model and returns the output.
        """
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x
