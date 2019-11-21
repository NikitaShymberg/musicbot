"""
This file contains the PrepData class that loads and preprocesses numpy arrays
TODO: redoc things
TODO: new tests
"""
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import tensorflow as tf

from constants import NUM_MEASURES, NUM_NOTES, NUM_TIMES,\
    PCA_DIMENSIONS, BATCH_SIZE, CONF_THRESH


class PrepData():
    """
    Loads all .npy files from the specified path and stores them
    as well as their principal components.
    This is the class that contains the train and test tensorflow datasets.
    """
    def __init__(self, data_path, save_path, num_songs):
        """
        Initializes a PrepData object to load .npy files from the
        `data_path` directory. The object will load at most `num_songs`
        songs.
        """
        self.data_path = data_path
        self.train_save_path = os.path.join(save_path, "train_songs")
        self.test_save_path = os.path.join(save_path, "test_songs")
        self.num_songs = num_songs
        self.pc_decomposer = IncrementalPCA(n_components=PCA_DIMENSIONS,
                                            batch_size=1000)

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

    def load_npy_file(self, file_name: str) -> np.ndarray:
        """
        Loads a single npy file from the given self.data_path/`file_name`.
        Returns all the (flattened) songs in the file.
        """
        songs = np.unpackbits(
                np.load(
                    os.path.join(self.data_path, file_name),
                ), axis=-1
            )
        return songs.reshape((songs.shape[0], -1))

    def fit_decomposer(self):
        """
        Loads songs from `self.data_path` and calls
        `self.pc_decomposer.partial_fit` for them all.
        """
        song_num = 0
        song_batch = None
        print("Fitting decomposer...")
        for file in tqdm(os.listdir(self.data_path)):
            if song_num >= self.num_songs:
                # All necessary songs loaded
                break
            songs = self.load_npy_file(file)
            if songs.shape[0] + song_num <= self.num_songs:
                # Not too many songs in `file`
                pass
                # self.pc_decomposer.partial_fit(songs)
            else:
                # Too many songs in `file`
                remaining_songs = self.num_songs - song_num
                songs = songs[:remaining_songs]
                # self.pc_decomposer.partial_fit(songs)
            if song_batch is not None:
                song_batch = np.concatenate((song_batch, songs))
            else:
                song_batch = songs
            if song_batch.shape[0] >= PCA_DIMENSIONS:
                # Fit the ipca
                self.pc_decomposer.partial_fit(song_batch)
                song_batch = None
            song_num += songs.shape[0]

    def serialize_song(self, pc: np.ndarray, song: np.ndarray) -> bytes:
        """
        Serializes the flattened `song` to be ready to be saved to a TFRecord.
        Taken from: https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        feature = {
            'pc': tf.train.Feature(
                float_list=tf.train.FloatList(value=pc)
            ),
            'song': tf.train.Feature(
                float_list=tf.train.FloatList(value=song)
            )
        }
        proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return proto.SerializeToString()

    def save_tf_records(self):
        """
        Loads songs from `self.data_path`, gets their principal components
        and saves them in a TFRecords in `self.train_save_path` and
        `self.test_save_path`.
        For each song there is a 20% chance that it gets put into the test set,
        thus the train/test split won't be an exact 80/20 split.
        Taken from: https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        song_num = 0
        print("Retrieving principal components...")
        # Train songs
        for file in tqdm(os.listdir(self.data_path)):
            if song_num >= self.num_songs:
                # All necessary songs loaded
                break
            songs = self.load_npy_file(file)
            if songs.shape[0] + song_num <= self.num_songs:
                # Not too many songs in `file`
                pc = self.pc_decomposer.transform(songs)
            else:
                # Too many songs in `file`
                remaining_songs = self.num_songs - song_num
                songs = songs[:remaining_songs]
                pc = self.pc_decomposer.transform(songs)
            song_num += songs.shape[0]
            for i in range(len(songs)):
                # Write TFRecord
                save_path = self.train_save_path if np.random.rand() < 0.8 \
                    else self.test_save_path
                with tf.io.TFRecordWriter(save_path) as writer:
                    example = self.serialize_song(pc[i], songs[i])
                    writer.write(example)

    def create_pcs(self, show_variance=False) -> None:
        """
        TODO: redoc
        Loads the data into `self.train_ds` and `self.test_ds`. Each of those
        datasets will contain tuples in the form (x, y) where x is the
        principle component of the song y.
        If `show_variance` is True, displays a plot showing the cumulative sum
        of the variance ratios of each principle component.
        """
        self.fit_decomposer()
        self.save_tf_records()

        if show_variance:
            import matplotlib.pyplot as plt
            plt.plot(np.cumsum(self.pc_decomposer.explained_variance_ratio_))
            plt.show()

        # self.create_datasets(pc, self.songs)

    def get_data(self, is_train=False):
        """
        Loads the TFRecord from `self.save_path` and creates a Dataset
        with the contents. Returns a generator that yields touples of
        (pc, song) values.
        FIXME: am broken?
        """
        load_path = self.train_save_path if is_train else self.test_save_path
        raw_dataset = tf.data.TFRecordDataset(
            load_path
        ).shuffle(self.num_songs).batch(BATCH_SIZE)
        for rr in raw_dataset:
            tf_eg = tf.train.Example.FromString(rr.numpy())
            pc = np.array(tf_eg.features.feature['pc'].float_list.value)
            song = np.array(tf_eg.features.feature['song'].float_list.value)
            yield pc, song


if __name__ == "__main__":
    d = PrepData("data/npy/", "data/tfrecords/", 100)
    d.create_pcs(show_variance=False)
    for q in d.get_data():
        print(q)
