import os
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class ThermoClass:
    """A class for defining which label is assigned to each class used in the classification step."""

    def __init__(self, class_name: str, class_value: int, class_folder: str = None):
        """Builds a class used for classification with the parameter passed as argument.

        :param class_name: Human readable name associated to this class.
        :param class_value: Numerical value (label) associated to this class.
        :param class_folder: Folder where the training images associated to this class are stored.
        """
        self.class_name = class_name
        self.class_value = class_value

        if class_folder is None:
            class_folder = self.class_name
        self.class_folder = class_folder


ThermoClassList = List[ThermoClass]


def create_directory_list(root_dir: str):
    """Creates a list of directories for dataset loading.

    :param root_dir: Absolute path to the root directory of the dataset.

    .. note:: The dataset root directory must be of the following form:
        ::

           root_dir
           |__video1
           |    |__0-1000
           |    |__1000_2000
           |__video2
           |    |__0-500
           |    |__500-1000
           |    |__1000-1200
           |__video3
                |__0-1000

        and each folder 'xxxx-yyyy' must contain three directories associated to the classes of the dataset (see :attr:`ThermoClass.class_folder`).

    :return: A list of absolute paths to the class directories containing the dataset images.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError("Directory {} does not exist".format(root_dir))

    # List all directories associated to different videos.
    recording_path_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    input_data_path = []
    for g in recording_path_list:
        # Append the different directories associated to different video frame intervals.
        input_data_path.extend([os.path.join(g, f) for f in os.listdir(g)])

    return input_data_path


class ThermoDataset:
    """Dataset class which handles the input image as a dataset.

    :Example:

        .. code-block:: python

            dataset = ThermoDataset(img_shape, batch_size)
            dataset.load_dataset(root_directory_list, class_list)

            train_iterator = dataset.get_train_iterator()
            next_train_batch = train_iterator.get_next()
            test_iterator = dataset.get_test_iterator()
            next_test_batch = test_iterator.get_next()

            # Build the computation graph.
            ...

            with tf.Session() as sess:
                for epoch in range(epochs):
                    sess.run(train_iterator.initializer)
                    sess.run(test_iterator.initializer)

                    # Train
                    while True:
                        try:
                            img_batch, label_batch = sess.run(next_train_batch)
                        except: # Training dataset is terminated
                            break
                        # Train the model.
                        ...

                    # Test
                     while True:
                        try:
                            img_batch, label_batch = sess.run(next_test_batch)
                        except: # Test dataset is terminated
                            break
                        # Test the model.

    """

    def __init__(self, img_shape: np.ndarray, batch_size: int = 32, balance_data: bool = True,
                 normalize_images: bool = True):
        """ Initializes the parameters of the dataset without loading anything.

        :param img_shape: Image shape of the dataset. All images on disk which don't fulfill this shape are resized accordingly.
        :param batch_size: Batch size used for training. This parameters influences the size of tha batch returned by the dataset iterators (see :func:`<self.get_train_iterator <ThermoDataset.get_train_iterator>`)
        :param balance_data: Boolean flag which determines whether to balance the data on disk. If True, the loaded classes will have the same amount of samples (some samples of the majority classes will be discarded).
        :param normalize_images: Boolean flag indicating whether no normalize each input image (mean: 0, std: 1)
        """
        self.image_shape = img_shape
        self.batch_size = batch_size
        self.balance_data = balance_data
        self.normalize_images = normalize_images

        self.__train_dataset = None
        self.__test_dataset = None
        self.__validation_dataset = None
        self.__root_directory_list = None
        self.__thermo_class_list = None
        self.num_classes = None
        self.__samples_per_class = []

        self.__train_fraction = 0.8
        self.__test_fraction = 0.2
        self.__validation_fraction = 0.0

        self.__image_file_names = None
        self.__labels = None

    def load_dataset(self, root_directory_list: list, class_list: ThermoClassList, load_all_data: bool = False) -> None:
        """Loads the dataset from the files contained in the list of root directories.

        :param root_directory_list: List of root directories containing the data. This list can be generated using :func:`create_directory_list`.
        :param class_list: List of classes used for classification.
        :param load_all_data: Boolean flag indicating whether to preload the entire dataset in memory once, or to load the data on the fly whenever a new batch is needed.

        .. note:: Loading the entire dataset into memory can increase the training process.
        """
        self.root_directory_list = root_directory_list
        self.thermo_class_list = class_list

        self.__image_file_names = np.array([], dtype=str)
        self.__labels = np.array([], dtype=np.int32)
        sample_per_class = {}
        for thermo_class in sorted(class_list, key=lambda t: t.class_value):
            for root_dir in self.root_directory_list:
                directory = os.path.join(root_dir, thermo_class.class_folder)
                image_names = np.array([os.path.join(directory, img_name) for img_name in os.listdir(directory)
                                        if img_name.endswith(".jpg")], dtype=str)
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

    def set_train_test_validation_fraction(self, train_fraction, test_fraction, validation_fraction) -> None:
        """Sets the train-test-validation fraction of the dataset."""
        total = train_fraction + test_fraction + validation_fraction
        self.__train_fraction = float(train_fraction) / total
        self.__test_fraction = float(test_fraction) / total
        self.__validation_fraction = float(validation_fraction) / total

    def get_train_iterator(self) -> tf.contrib.data.Iterator:
        """Builds and returns an initializable iterator for the training dataset."""
        return self.train.make_initializable_iterator()

    def get_test_iterator(self) -> tf.contrib.data.Iterator:
        """Builds and returns an initializable iterator for the test dataset."""
        return self.test.make_initializable_iterator()

    def get_validation_iterator(self) -> tf.contrib.data.Iterator:
        """Builds and returns an initializable iterator for the validation dataset."""
        return self.validation.make_initializable_iterator()

    def print_info(self):
        """Prints the dataset properties."""
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

    @property
    def image_shape(self) -> np.ndarray:
        """Returns the image shape of the dataset."""
        return self.__image_shape

    @image_shape.setter
    def image_shape(self, l: np.ndarray):
        if len(l) != 3:
            raise ValueError("Image shape passed to dataset must be of length 3! Passed: {}".format(l))
        self.__image_shape = l

    @property
    def rgb(self) -> bool:
        """Returns a boolean indicating whether the dataset has three channels (RGB) or is grayscale."""
        return self.image_shape[2] == 3

    @property
    def data_size(self) -> int:
        """Returns the size of the dataset, i.e. the total number of images loaded."""
        return len(self.__labels)

    @property
    def train_size(self) -> int:
        """Returns the size of the training data, i.e. the total number of images available for training."""
        return int(self.data_size * self.__train_fraction)

    @property
    def test_size(self) -> int:
        """Returns the size of the testing data, i.e. the total number of images available for testing."""
        return int(self.data_size * self.__test_fraction)

    @property
    def validation_size(self) -> int:
        """Returns the size of the validation data, i.e. the total number of images available for validation."""
        return int(self.data_size * self.__validation_fraction)

    @property
    def train(self) -> tf.contrib.data.Dataset:
        """Returns a reference to the training data."""
        return self.__train_dataset

    @property
    def test(self) -> tf.contrib.data.Dataset:
        """Returns a reference to the test data."""
        return self.__test_dataset

    @property
    def validation(self) -> tf.contrib.data.Dataset:
        """Returns a reference to the validation data."""
        return self.__validation_dataset

    @property
    def split_fraction(self):
        """Returns the fraction used to split the loaded data into train, test and validation data."""
        return np.array([self.__train_fraction, self.__test_fraction, self.__validation_fraction])

    @property
    def root_directory_list(self) -> str:
        """Returns the list of root directories of the data."""
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
    def thermo_class_list(self) -> ThermoClassList:
        """Returns the classes associated to the dataset."""
        if self.__thermo_class_list is None:
            raise ValueError("Property 'thermo_class_list' has not been set yet!")
        return self.__thermo_class_list

    @thermo_class_list.setter
    def thermo_class_list(self, thermo_class_list: ThermoClassList):
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

    def __balance_data(self):
        """Balances the data such that all classes are represented by the same amount of sample. This will discard the remaining samples of the majority classes."""
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

    def __parse_image(self, image_path: str, image_label: int) -> tuple:
        """Function used by tensorflow to preprocess the data when loaded at runtime.

        :param image_path: absolute path to the input image.
        :param image_label: integer associated to the input image label.
        :return: A tuple consisting of the preprocessed image, and the one-hot encoding of the label.
        """
        one_hot = tf.one_hot(image_label, self.num_classes, dtype=dtypes.int32)
        img_file = tf.read_file(image_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=self.image_shape[2])
        img_decoded = tf.image.resize_images(img_decoded, self.image_shape[0:2])
        img_decoded = tf.cast(img_decoded, tf.float32)
        if self.normalize_images:
            img_decoded = tf.image.per_image_standardization(img_decoded)

        return img_decoded, one_hot

    def __parse_image_load(self, image_path: str, image_label: int):
        """Function used to preprocess the data when loaded all at once.

        :param image_path: absolute path to the input image.
        :param image_label: integer associated to the input image label.
        :return: A tuple consisting of the preprocessed image, and the one-hot encoding of the label.
        """
        one_hot = tf.one_hot(image_label, self.num_classes, dtype=dtypes.int32)
        if self.rgb:
            flag = cv2.IMREAD_COLOR
        else:
            flag = cv2.IMREAD_GRAYSCALE

        img = cv2.imread(image_path, flags=flag)
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_AREA).astype(
            np.float32)

        if self.normalize_images:
            img_mean = np.mean(img, axis=(0, 1))
            img_std = np.std(img, axis=(0, 1))

            img = (img - img_mean) / img_std

        return img, one_hot

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
                if not self.rgb:
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
