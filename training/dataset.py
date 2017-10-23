import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import numpy as np

from .thermo_class import ThermoClassList

import os


class ThermoDataset:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

        self.__train_dataset = None
        self.__test_dataset = None
        self.__validation_dataset = None
        self.__root_directory = None
        self.__thermo_class_list = None
        self.num_classes = None
        self.__samples_per_class = []

        self.__train_fraction = 0.6
        self.__test_fraction = 0.2
        self.__validation_fraction = 0.2

        self.__image_file_names = None
        self.__labels = None

    def get_class_weights(self):
        weights = np.array([1.0 / v for v in self.__samples_per_class], dtype=np.float32)
        weights /= np.sum(weights)
        return weights

    @property
    def data_size(self):
        return len(self.__labels)

    @property
    def train(self):
        return self.__train_dataset

    @property
    def test(self):
        return self.__test_dataset

    @property
    def validation(self):
        return self.__validation_dataset

    @property
    def split_fraction(self):
        return np.array([self.__train_fraction, self.__test_fraction, self.__validation_fraction])

    def dataset_from_id(self, index):
        return {0: self.__train_dataset, 1: self.__test_dataset, 2: self.__validation_dataset}[index]

    @property
    def root_directory(self):
        return self.__root_directory

    @root_directory.setter
    def root_directory(self, path: str):
        if not os.path.exists(path):
            raise ValueError("Path <{}> passed to 'root_directory' property does not exist".format(path))
        self.__root_directory = path

    @property
    def thermo_class_list(self):
        if self.__thermo_class_list is None:
            raise ValueError("Property 'thermo_class_list' has not been set yet!")
        return self.__thermo_class_list

    @thermo_class_list.setter
    def thermo_class_list(self, thermo_class_list):
        if self.__root_directory is None:
            raise ValueError("Must set property 'root_directory' before setting the class list!")
        directories_which_must_be_contained_in_root_directory = [thermo_class.class_name for thermo_class in
                                                                 thermo_class_list]

        for directory in directories_which_must_be_contained_in_root_directory:
            if directory not in os.listdir(self.root_directory):
                raise (
                    ValueError(
                        "Root directory {} does not contain subdirectory {}".format(self.root_directory, directory)))

        self.num_classes = len(thermo_class_list)
        thermo_class_labels = [thermo_class.class_value for thermo_class in thermo_class_list]
        for class_label in range(self.num_classes):
            if class_label not in thermo_class_labels:
                raise ValueError(
                    "Class label {} is not present in thermo classes: {}".format(class_label, thermo_class_labels))

    def load_dataset(self, root_directory: str, class_list: ThermoClassList):
        self.root_directory = root_directory
        self.thermo_class_list = class_list

        self.__image_file_names = np.array([], dtype=str)
        self.__labels = np.array([], dtype=np.int32)
        sample_per_class = {}
        for thermo_class in class_list:
            directory = os.path.join(root_directory, thermo_class.class_folder)
            image_names = np.array([os.path.join(directory, img_name) for img_name in os.listdir(directory)], dtype=str)
            self.__image_file_names = np.concatenate((self.__image_file_names, image_names))
            self.__labels = np.concatenate(
                (self.__labels, np.ones(shape=(len(image_names)), dtype=np.int32) * thermo_class.class_value))
            sample_per_class[thermo_class.class_value] = len(image_names)
        self.__samples_per_class = [sample_per_class[thermo_class.class_value] for thermo_class in class_list]

        permutation = np.random.permutation(len(self.__image_file_names))
        self.__image_file_names = self.__image_file_names[permutation]
        self.__labels = self.__labels[permutation]

        self.__create_internal_dataset()

    def set_train_test_validation_fraction(self, train_fraction, test_fraction, validation_fraction):
        total = train_fraction + test_fraction + validation_fraction
        self.__train_fraction = float(train_fraction) / total
        self.__test_fraction = float(test_fraction) / total
        self.__validation_fraction = float(validation_fraction) / total

    def __parse_image(self, image_path: str, image_label: int):
        one_hot = tf.one_hot(image_label, self.num_classes, dtype=dtypes.int32)
        img_file = tf.read_file(image_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        img_decoded = tf.image.rgb_to_grayscale(img_decoded)
        img_decoded = tf.image.resize_images(img_decoded, [24, 30])

        return img_decoded, one_hot

    def __create_internal_dataset(self):
        cumulative_fraction = 0.0
        for dataset_id in range(3):
            fraction = self.split_fraction[dataset_id]
            min_index = int(np.floor(cumulative_fraction * self.data_size))
            max_index = int(np.floor((cumulative_fraction + fraction) * self.data_size))
            cumulative_fraction += fraction

            images = convert_to_tensor(self.__image_file_names[min_index:max_index], dtypes.string)
            labels = convert_to_tensor(self.__labels[min_index:max_index], dtypes.int32)

            data = Dataset.from_tensor_slices((images, labels))
            data = data.map(self.__parse_image)

            # Create a new dataset with batches of images
            data = data.batch(self.batch_size)
            if dataset_id == 0:
                self.__train_dataset = data
            elif dataset_id == 1:
                self.__test_dataset = data
            else:
                self.__validation_dataset = data

    def get_train_iterator(self):
        return self.train.make_initializable_iterator()

    def get_test_iterator(self):
        return self.test.make_initializable_iterator()

    def get_validation_iterator(self):
        return self.validation.make_initializable_iterator()

    def print_info(self):
        print("Num samples (train/test/val):  {} tot: {}\n"
              "Sample type   {}\n"
              "Sample shape: {}\n"
              "Label type    {}\n"
              "Label shape:  {}\n"
              "Root dir:     {}".format([int(np.floor(frac * len(self.__labels))) for frac in self.split_fraction],
                                        len(self.__labels),
                                        self.train.output_types[0], self.train.output_shapes[0][1:],
                                        self.train.output_types[1], self.train.output_shapes[1][1:],
                                        self.__root_directory))
