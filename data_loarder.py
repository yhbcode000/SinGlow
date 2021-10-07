import os
import pickle

import librosa
from tqdm import tqdm

from common_definitions import *


class SongDataLoader:
    """
    SongDataLoader object
    1. load tfrecord dataset
    2. make tfrecord dataset
    """

    def __init__(self, *tfrecord_names, tfrecord_dir):
        # Step 1. the data, split between train and test sets
        self.tfrecord_dir = tfrecord_dir
        if not os.path.exists(self.tfrecord_dir):
            os.mkdir(tfrecord_dir)
        self.tfrecord_names = tfrecord_names
        self.tfrecord_files_dir = [os.path.join(self.tfrecord_dir, tfrecord_name) for tfrecord_name in
                                   self.tfrecord_names]
        self.batch_number = None

    def make(self, *files_dir):
        """
        convert mp3 files dir to tfrecord
        """
        assert len(files_dir) == len(self.tfrecord_files_dir)
        for file_dir, tfrecord_file_dir in zip(files_dir, self.tfrecord_files_dir):
            if os.path.exists(tfrecord_file_dir):
                continue
            # load original data
            song_data = np.array(
                [librosa.load(os.path.join(file_dir, f))[0] for f in tqdm(os.listdir(file_dir)) if
                 f[-4:] in ['.mp3', '.wav']])

            # write to tfrecord file
            with tf.io.TFRecordWriter(tfrecord_file_dir) as writer:
                for data in tqdm(song_data):
                    shape = data.shape
                    sound = data.tobytes()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0]])),
                        'sound': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sound]))}))

                    writer.write(example.SerializeToString())

    def load(self, sampling_num):
        """
        Load data
        Returns: tf dataset: train dataset
        """
        dataset = tf.data.TFRecordDataset(self.tfrecord_files_dir)
        dataset = dataset.map(map_func=self.__extract_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = self.__dataset_trans(dataset, sampling_num=sampling_num)
        pickle_name = f'./{"_".join(self.tfrecord_names)}.pickle'
        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as f:
                d = pickle.load(f)
        else:
            d = []
            for i in tqdm(dataset):
                d += [i.numpy()]
            with open(pickle_name, 'wb') as f:
                pickle.dump(d, f)
        self.batch_number = len(d) // BATCH_SIZE + 1
        song_dataset = tf.data.Dataset.from_tensor_slices(d).shuffle(buffer_size=BUFFER_SIZE).batch(
            BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        return song_dataset

    def __extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        features = {
            'shape': tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
            'sound': tf.io.FixedLenFeature([], tf.string), }

        # Extract the data record
        sample = tf.io.parse_single_example(tfrecord, features)

        shape = sample["shape"][0]
        sound = tf.io.decode_raw(sample['sound'], tf.float32)

        sound = tf.reshape(sound, shape=(shape, 1, CHANNEL_SIZE), name=None)

        sound = tf.image.resize(sound, [shape // 22050 * SAMPLING_RATE, 1])

        return sound, tf.shape(sound)[0]

    def __take_random_window(self, sound, shape):
        rand = tf.random.uniform(shape=(), minval=0, maxval=shape - WINDOW_LENGTH - 1, dtype=tf.int32)
        cropped = sound[rand:rand + WINDOW_LENGTH]
        return cropped

    def __take_random_aligned_window(self, sound, shape):
        rand = tf.random.uniform(shape=(), minval=0, maxval=shape // WINDOW_LENGTH - 1, dtype=tf.int32)
        cropped = sound[rand * WINDOW_LENGTH:(rand + 1) * WINDOW_LENGTH]
        return cropped

    def __dataset_trans(self, dataset, sampling_num):
        dataset_after = dataset.map(map_func=self.__take_random_aligned_window,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for i in range(sampling_num - 1):
            dataset_after = dataset_after.concatenate(
                dataset.map(map_func=self.__take_random_aligned_window, num_parallel_calls=tf.data.experimental.AUTOTUNE))
        return dataset_after
