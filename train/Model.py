"""
This class contains the music generation model
"""
import tensorflow as tf
from constants import PCA_DIMENSIONS, NUM_MEASURES, NUM_NOTES, NUM_TIMES


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
        self.d1 = tf.keras.layers.Dense(PCA_DIMENSIONS * 2, activation='relu')
        self.d2 = tf.keras.layers.Dense(PCA_DIMENSIONS * 4, activation='relu')
        self.d3 = tf.keras.layers.Dense(NUM_MEASURES * NUM_NOTES * NUM_TIMES,
                                        activation='relu')

    def call(self, x):
        """
        Runs the model on input `x`.
        """
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x
