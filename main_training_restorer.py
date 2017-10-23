import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Iterator

from training.dataset import ThermoDataset
from training.models.simple_net import SimpleNet
from training.thermo_class import ThermoClass

path = "C:/Users/Carlo/Desktop/Ghidoni"
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(path, "tensorboard")
checkpoint_path = os.path.join(path, "checkpoints")

batch_size = 100
num_classes = 3

path = "C:/Users/Carlo/Desktop/Ghidoni"

working_class = ThermoClass("working", 0)
broken_class = ThermoClass("broken", 1)
misdetected_class = ThermoClass("misdetected", 2)
thermo_class_list = [working_class, broken_class, misdetected_class]

if __name__ == '__main__':
    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        with tf.name_scope("dataset"):
            with tf.name_scope("loading"):
                dataset = ThermoDataset(batch_size=batch_size, shuffle=True, buffer_size=10)
                dataset.load_dataset(root_directory=path, class_list=thermo_class_list)

            with tf.name_scope("iterator"):
                # create an reinitializable iterator given the dataset structure
                iterator = Iterator.from_structure(dataset.train.output_types,
                                                   dataset.train.output_shapes)
                next_batch = iterator.get_next()

                # Ops for initializing the two different iterators
                data_init_op = iterator.make_initializer(dataset.train)

    with tf.name_scope("placeholders"):
        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [batch_size, 24, 30, 1], name="input_image")
        y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_probab")

    model = SimpleNet(x=x, keep_prob=0.8, num_classes=num_classes)

    predict_op = tf.argmax(model.logits, axis=1, name="model_predictions")

    correct_pred = tf.equal(predict_op, tf.argmax(y, 1), name="correct_predictions")
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    restorer = tf.train.Saver()
    with tf.Session() as sess:
        print("Opened session")
        sess.run(data_init_op)
        print("Runned data init op")

        # Restore variables from disk.
        checkpoint_name = os.path.join(checkpoint_path, "simple_model-25")
        print("Loading {}".format(checkpoint_name))
        restorer.restore(sess, checkpoint_name)
        print("Restored")
        # sess.run(tf.initialize_all_variables())

        correct = 0
        wrong = 0

        confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int32)

        for i in range(10):
            print(i)
            s = time.time()
            img, lab = sess.run(next_batch)
            ss = time.time()
            print("Fetched batch in {} s".format(ss - s))
            acc, pred, logits = sess.run([accuracy_op, predict_op, model.logits], feed_dict={x: img, y: lab, keep_prob: 1.0})
            print("Inference in {} s".format(time.time() - ss))

            l = np.argmax(lab, axis=1)
            p = pred

            for ll, pp in zip(l, p):
                confusion_matrix[ll, pp] += 1

                # img = np.squeeze(img[0])
                # print(np.int32(img))
                # print("min_value = {}, max_value = {}".format(np.min(img, axis=(0,1)), np.max(img, axis=(0,1))))
                # img = np.uint8(img)
                # print(img.shape)
                # img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_AREA)
                # cv2.imshow("True: {}, Pred: {}, logits: {}".format(np.argmax(lab), pred, sess.run(tf.nn.softmax(logits))),
                #            img)
                # cv2.waitKey(0)


        print("Confusion matrix:\n{}".format(confusion_matrix))
        correct = np.sum(np.diag(confusion_matrix))
        wrong = np.sum(confusion_matrix, axis=(0, 1)) - correct
        print("Correct predictions: {}\nWrong predictions: {}\nAccuracy: {}".format(correct, wrong,
                                                                                    float(correct) / (correct + wrong)))
        print("Accuracy op: {}".format(acc))
