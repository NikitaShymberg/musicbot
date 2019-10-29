"""
This file contains the PrepData class that loads and preprocesses numpy arrays
"""
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from constants import NUM_MEASURES, NUM_NOTES, NUM_TIMES, PCA_DIMENSIONS


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
        Loads the data into `self.songs` and the principal components into `self.pc`.
        """
        song_num = 0
        for file in tqdm(os.listdir(self.data_path)):
            songs = np.unpackbits(
                    np.load(
                        os.path.join(self.data_path, file)
                    ), axis=-1
                )
            for cur_song in songs:
                if song_num >= self.num_songs:
                    break
                self.songs[song_num] = cur_song
                song_num += 1
        decomposer = PCA(n_components=PCA_DIMENSIONS).fit(self.songs.reshape((self.num_songs, -1)))
        # plt.plot(np.cumsum(decomposer.explained_variance_ratio_))
        # plt.plot(decomposer.explained_variance_ratio_)
        # plt.show()
        self.pc = decomposer.transform(self.songs.reshape((self.num_songs, -1)))


if __name__ == "__main__":
    d = PrepData("data/npy/", 1000)
    # d = PrepData("data/npy/", 22229)
    d.load_data()
