"""
This file tests the PrepData class
"""
import unittest
import numpy as np

from train.PrepData import PrepData
from constants import NUM_TIMES, NUM_MEASURES, NUM_NOTES, BATCH_SIZE


class TestPrepData(unittest.TestCase):
    """
    Tests the PrepData class
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.data_path = "path"
        self.num_songs = BATCH_SIZE * 2 + 1
        self.flat_song = np.zeros((NUM_MEASURES * NUM_NOTES * NUM_TIMES, ),
                                  dtype=np.float)
        self.pcs = np.random.rand(self.num_songs, 100)
        self.songs = np.random.rand(self.num_songs, 100)

    def setUp(self):
        """
        Creates a PrepData object to be used by other tests.
        """
        self.pd = PrepData(self.data_path, self.num_songs)

    def test_constructor_songs(self):
        """
        Ensures that the constructor creates the self.songs
        attribute correctly.
        """
        self.assertEqual(self.pd.songs.shape,
                         (self.num_songs, NUM_MEASURES, NUM_TIMES, NUM_NOTES))

    def test_to_song_shape(self):
        """
        Ensures that the to_song method reshapes the song correctly.
        """
        song = PrepData.to_song(self.flat_song)
        self.assertEqual(song.shape, (NUM_MEASURES, NUM_TIMES, NUM_NOTES))

    def test_to_song_dtype(self):
        """
        Ensures that the to_song method sets the data
        type of the song correctly.
        """
        song = PrepData.to_song(self.flat_song)
        self.assertEqual(song.dtype, np.int8)

    def test_create_dataset_train_test(self):
        """
        Ensures that the create_dataset method creates train and test datasets.
        """
        self.pd.create_datasets(self.pcs, self.songs)
        shuffled_elements_train = [x for x in self.pd.train_ds][0]
        shuffled_elements_test = [x for x in self.pd.test_ds][0]
        self.assertGreater(len(self.pcs), len(shuffled_elements_train[0]))
        self.assertGreater(len(self.pcs), len(shuffled_elements_test[0]))

    def test_create_dataset_shuffle(self):
        """
        Ensures that the create_dataset method shuffles the train dataset.
        NOTE: this test will fail if some data element's position
        remains unchanged - FIXME.
        """
        self.pd.create_datasets(self.pcs, self.songs)
        shuffled_elements = [x for x in self.pd.train_ds][0]
        og_elements = self.pcs[:len(shuffled_elements)]
        for i in range(len(shuffled_elements)):
            shuffled_x = shuffled_elements[0][i].numpy()
            self.assertTrue((og_elements[i] != shuffled_x).all())

    def test_create_dataset_batch_size(self):
        """
        Ensures that the create_dataset method batches
        the datasets to BATCH_SIZE batches.
        """
        self.pd.create_datasets(self.pcs, self.songs)
        for x in self.pd.train_ds:
            self.assertEqual(x[0].shape[0], BATCH_SIZE)
            break

    def test_load_songs_saves_memory(self):
        """
        Ensures that the load_songs method deletes self.songs.
        """
        ...

    def test_load_songs_loads_n_songs(self):
        """
        Ensures that the load_songs method only the correct number of songs.
        """
        ...
