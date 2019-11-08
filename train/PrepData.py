"""
This file contains the PrepData class that loads and preprocesses numpy arrays
"""
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

from constants import NUM_MEASURES, NUM_NOTES, NUM_TIMES,\
    PCA_DIMENSIONS, BATCH_SIZE


class PrepData():
    """
    Loads all .npy files from the specified path and stores them
    as well as their principal components.
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

    def load_data(self):
        """
        Loads the data into `self.train_songs` and `self.test_songs`
        and the principal components into `self.train_pc` and `self.train_pc`.
        TODO: refactor
        """
        song_num = 0
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
        decomposer = TruncatedSVD(n_components=PCA_DIMENSIONS).fit(self.songs)
        pc = decomposer.transform(self.songs)

        # import matplotlib.pyplot as plt
        # plt.plot(np.cumsum(decomposer.explained_variance_ratio_))
        # plt.show()

        # Split into random train test subsets
        indices = np.arange(self.num_songs)
        np.random.shuffle(indices)
        num_train = int(self.num_songs * 0.8)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (pc[train_indices], self.songs[train_indices])
        ).shuffle(num_train).batch(BATCH_SIZE)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (pc[test_indices], self.songs[test_indices])
        ).shuffle(self.num_songs - num_train).batch(BATCH_SIZE)
        del self.songs


if __name__ == "__main__":
    d = PrepData("data/npy/", 2294)
    # d = PrepData("data/npy/", 22229)
    d.load_data()
    print("Done!")
