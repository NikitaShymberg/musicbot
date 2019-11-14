"""
This file contains the PrepData class that loads and preprocesses numpy arrays
"""
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

from constants import NUM_MEASURES, NUM_NOTES, NUM_TIMES,\
    PCA_DIMENSIONS, BATCH_SIZE, CONF_THRESH


class PrepData():
    """
    Loads all .npy files from the specified path and stores them
    as well as their principal components.
    This is the class that contains the train and test tensorflow datasets.
    """
    def __init__(self, data_path, num_songs):
        """
        Initializes a PrepData object to load .npy files from the
        `data_path` directory. The object will load at most `num_songs`
        songs.
        """
        self.data_path = data_path
        self.num_songs = num_songs
        self.songs = np.zeros((num_songs, NUM_MEASURES, NUM_TIMES, NUM_NOTES),
                              dtype=np.int8)

    @staticmethod
    def to_song(song: np.ndarray) -> np.ndarray:
        """
        Used to convert a flattened float song to a song shape and data type.
        Returns the song with shape (NUM_MEASURES, NUM_TIMES, NUM_NOTES)
        and int8 dtype. Notes whose confidence level is above CONF_THRESH
        are considered to exist, others are discarded.
        """
        song = np.copy(song)
        song[song >= CONF_THRESH] = 1
        song[song < CONF_THRESH] = 0
        return song.reshape(
            (NUM_MEASURES, NUM_TIMES, NUM_NOTES)
        ).astype(np.int8)

    def create_datasets(self, pcs: np.ndarray, songs: np.ndarray) -> None:
        """
        Creates and stores the `self.train_ds` and `self.test_ds` datasets
        taking a randomly selected 20% for the test set.
        It is important that the indices of each pc match up to the song they
        were derived from.
        """
        # Shuffle
        indices = np.arange(self.num_songs)
        np.random.shuffle(indices)
        num_train = int(self.num_songs * 0.8)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        # Create datasets, I think the additional call to shuffle
        # will shuffle the dataset whenever we finish iterating throuygh it
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (pcs[train_indices], songs[train_indices])
        ).shuffle(num_train).batch(BATCH_SIZE)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (pcs[test_indices], songs[test_indices])
        ).shuffle(self.num_songs - num_train).batch(BATCH_SIZE)

    def load_data(self, show_variance=False) -> None:
        """
        Loads the data into `self.train_ds` and `self.test_ds`. Each of those
        datasets will contain tuples in the form (x, y) where x is the
        principle component of the song y.
        If `show_variance` is True, displays a plot showing the cumulative sum
        of the variance ratios of each principle component.
        """
        # Load songs
        song_num = 0
        print("Loading data...")
        for file in tqdm(os.listdir(self.data_path)):
            songs = np.unpackbits(
                    np.load(
                        os.path.join(self.data_path, file),
                    ), axis=-1
                )
            for cur_song in songs:
                if song_num >= self.num_songs:
                    break
                self.songs[song_num] = cur_song
                song_num += 1
        self.songs = self.songs.reshape((self.num_songs, -1))

        # Get principle components
        print("Retrieving principle components...")
        decomposer = TruncatedSVD(n_components=PCA_DIMENSIONS).fit(self.songs)
        pc = decomposer.transform(self.songs)

        if show_variance:
            import matplotlib.pyplot as plt
            plt.plot(np.cumsum(decomposer.explained_variance_ratio_))
            plt.show()

        self.create_datasets(pc, self.songs)
        del self.songs


if __name__ == "__main__":
    d = PrepData("data/npy/", 2294)
    d.load_data()
    print("Done!")
