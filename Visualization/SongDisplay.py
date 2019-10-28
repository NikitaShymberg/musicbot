"""
This file contains the SongDisplay class
"""

import matplotlib.pyplot as plt
import numpy as np

from constants import NUM_MEASURES


class SongDisplay():
    @staticmethod
    def show(song: np.ndarray):
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(1, NUM_MEASURES + 1):
            fig.add_subplot(4, 4, i)
            plt.imshow(song[i-1].T)
            plt.gca().invert_yaxis()
            plt.axis('off')
        plt.show()


if __name__ == "__main__":
    bk1 = np.load("data/temp/bak1.mid.npy")
    bk2 = np.load("data/temp/bak2.mid.npy")
    SongDisplay.show(bk1[0])
    SongDisplay.show(bk2[0])
