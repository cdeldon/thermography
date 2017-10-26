import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import numpy as np
import cv2

from .thermo_class import ThermoClassList

import os


class ThermoDataset:
    def __init__(self, image_shape: list, batch_size: int = 32, balance_data=True):
        self.image_shape = image_shape
        assert (len(self.image_shape) == 2)
        self.batch_size = batch_size
        self.balance_data = balance_data

        self.__train_dataset = None
        self.__test_dataset = None
        self.__validation_dataset = None
        self.__root_directory_list = None
        self.__thermo_class_list = None
        self.num_classes = None
        self.__samples_per_class = []

        self.__train_fraction = 0.6
        self.__test_fraction = 0.2
        self.__validation_fraction = 0.2

        self.__image_file_names = None
        self.__labels = None

    @property
    def data_size(self):
        return len(self.__labels)

    @property
    def train_size(self):
        return int(self.data_size * self.__train_fraction)

    @property
    def test_size(self):
        return int(self.data_size * self.__test_fraction)

    @property
    def validation_size(self):
        return int(self.data_size * self.__validation_fraction)

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
    def root_directory_list(self):
        return self.__root_directory_list

    @root_directory_list.setter
    def root_directory_list(self, dir_list: list):
        if type(dir_list) is str:
            dir_list = [dir_list]
        for directory in dir_list:
            if not os.path.exists(directory):
                raise ValueError(
                    "Directory <{}> passed to 'root_directory_list' property does not exist".format(directory))
        self.__root_directory_list = dir_list

    @property
    def thermo_class_list(self):
        if self.__thermo_class_list is None:
            raise ValueError("Property 'thermo_class_list' has not been set yet!")
        return self.__thermo_class_list

    @thermo_class_list.setter
    def thermo_class_list(self, thermo_class_list):
        if self.__root_directory_list is None:
            raise ValueError("Must set property 'root_directory_list' before setting the class list!")
        directories_which_must_be_contained_in_root_directory = [thermo_class.class_name for thermo_class in
                                                                 thermo_class_list]

        for directory in directories_which_must_be_contained_in_root_directory:
            for dir in self.__root_directory_list:
                if directory not in os.listdir(dir):
                    raise (ValueError("Root directory {} does not contain subdirectory {}".format(dir, directory)))

        self.num_classes = len(thermo_class_list)
        thermo_class_labels = [thermo_class.class_value for thermo_class in thermo_class_list]
        for class_label in range(self.num_classes):
            if class_label not in thermo_class_labels:
                raise ValueError(
                    "Class label {} is not present in thermo classes: {}".format(class_label, thermo_class_labels))

    def load_dataset(self, root_directory_list: list, class_list: ThermoClassList, load_all_data: bool = False):
        self.root_directory_list = root_directory_list
        self.thermo_class_list = class_list

        self.__image_file_names = np.array([], dtype=str)
        self.__labels = np.array([], dtype=np.int32)
        sample_per_class = {}
        for thermo_class in sorted(class_list, key=lambda t: t.class_value):
            for root_dir in self.root_directory_list:
                directory = os.path.join(root_dir, thermo_class.class_folder)
                image_names = np.array([os.path.join(directory, img_name) for img_name in os.listdir(directory)],
                                       dtype=str)
                self.__image_file_names = np.concatenate((self.__image_file_names, image_names))
                self.__labels = np.concatenate(
                    (self.__labels, np.ones(shape=(len(image_names)), dtype=np.int32) * thermo_class.class_value))
                if thermo_class.class_value not in sample_per_class:
                    sample_per_class[thermo_class.class_value] = len(image_names)
                else:
                    sample_per_class[thermo_class.class_value] += len(image_names)

        self.__samples_per_class = [sample_per_class[thermo_class.class_value] for thermo_class in class_list]

        if self.balance_data:
            self.__balance_data()

        permutation = np.random.permutation(len(self.__image_file_names))
        self.__image_file_names = self.__image_file_names[permutation]
        self.__labels = self.__labels[permutation]

        self.__create_internal_dataset(load_all_data)

    def __balance_data(self):
        # Shuffle each class independently (This is useful in case of multiple root directories because it does not
        # discard only elements of the last listed root directory, but random elements of all root directories)
        start_index = 0
        for class_id, num_samples_in_this_class in enumerate(self.__samples_per_class):
            permutation = np.random.permutation(num_samples_in_this_class)
            self.__image_file_names[start_index:start_index + num_samples_in_this_class] = \
                self.__image_file_names[start_index:start_index + num_samples_in_this_class][permutation]
            start_index += num_samples_in_this_class

        class_with_min_samples = np.argmin(self.__samples_per_class)
        num_min_samples = self.__samples_per_class[class_with_min_samples]

        # Remove all elements in the majority classes in order to balance their sample numbers to the minority class.
        start_index = 0
        elements_to_delete = []
        for num_samples_in_this_class in self.__samples_per_class:
            new_indices_to_delete = [i for i in
                                     range(start_index + num_min_samples, start_index + num_samples_in_this_class)]
            elements_to_delete.extend(new_indices_to_delete)
            start_index += num_samples_in_this_class

        self.__labels = np.delete(self.__labels, elements_to_delete)
        self.__image_file_names = np.delete(self.__image_file_names, elements_to_delete)

        # Check for class balance.
        cumulator = np.zeros(shape=3)
        for label in self.__labels:
            cumulator[label] += 1
        for i in range(2):
            if cumulator[i] != cumulator[i + 1]:
                raise RuntimeError("Error in data balancing: resulting label distribution: {}".format(cumulator))

        self.__samples_per_class = [num_min_samples for _ in range(self.num_classes)]

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
        img_decoded = tf.image.resize_images(img_decoded, self.image_shape)

        return img_decoded, one_hot

    def __parse_image_load(self, image_path: str, image_label: int):
        one_hot = tf.one_hot(image_label, self.num_classes, dtype=dtypes.int32)
        img_file = cv2.imread(image_path)
        img_decoded = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
        img_decoded = cv2.resize(img_decoded, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_AREA)
        img_decoded = cv2.normalize(img_decoded.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        return img_decoded, one_hot

    def __create_internal_dataset(self, load_all_data: bool):
        cumulative_fraction = 0.0
        for dataset_id in range(3):
            fraction = self.split_fraction[dataset_id]
            min_index = int(np.floor(cumulative_fraction * self.data_size))
            max_index = int(np.floor((cumulative_fraction + fraction) * self.data_size))
            cumulative_fraction += fraction

            if load_all_data:
                images = []
                labels = []
                num_images = max_index - min_index - 1
                print("Loading {} images for {} dataset.".format(num_images,
                                                                 {0: "TRAIN", 1: "TEST", 2: "VALIDAT."}[dataset_id]))
                for image_num, image_index in enumerate(range(min_index, max_index)):
                    image_path = self.__image_file_names[image_index]
                    image_label = self.__labels[image_index]
                    if (image_num + 1) % 100 == 0:
                        print("Loaded {} images of {}".format(image_num + 1, num_images))
                    im, l = self.__parse_image_load(image_path, image_label)
                    images.append(im)
                    labels.append(l)
                print("Loaded all {} images".format({0: "TRAIN", 1: "TEST", 2: "VALIDAT."}[dataset_id]))
                images = np.array(images)
                images = images[..., np.newaxis]
                print("Images shape: {}".format(images.shape))
                images = convert_to_tensor(images, dtypes.float32)
                labels = convert_to_tensor(labels, dtypes.int32)
            else:
                images = convert_to_tensor(self.__image_file_names[min_index:max_index], dtypes.string)
                labels = convert_to_tensor(self.__labels[min_index:max_index], dtypes.int32)

            data = Dataset.from_tensor_slices((images, labels))
            if not load_all_data:
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
              "Samples per class: {}\n"
              "Sample type        {}\n"
              "Sample shape:      {}\n"
              "Label type         {}\n"
              "Label shape:       {}\n"
              "Root dirs:         {}".format([int(np.floor(frac * len(self.__labels))) for frac in self.split_fraction],
                                             len(self.__labels),
                                             self.__samples_per_class,
                                             self.train.output_types[0], self.train.output_shapes[0][1:],
                                             self.train.output_types[1], self.train.output_shapes[1][1:],
                                             self.__root_directory_list))
