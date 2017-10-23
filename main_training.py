import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from training.dataset import ThermoDataset
from training.models import SimpleNet, ComplexNet
from training.thermo_class import ThermoClass


def main():
    input_data_path = "C:/Users/Carlo/Desktop/Ghidoni"

    working_class = ThermoClass("working", 0)
    broken_class = ThermoClass("broken", 1)
    misdetected_class = ThermoClass("misdetected", 2)
    thermo_class_list = [working_class, broken_class, misdetected_class]

    # Learning params
    num_epochs = 10000
    batch_size = 64
    learning_rate = 0.0025

    # Network params
    keep_probability = 0.8

    # Summary params
    write_train_summaries_every_n_steps = 10
    write_test_summaries_every_n_epochs = 1
    save_model_every_n_steps = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    summary_path = os.path.join(input_data_path, "tensorboard")
    checkpoint_path = os.path.join(input_data_path, "checkpoints")

    # Place data loading and preprocessing on the cpu.
    with tf.device('/cpu:0'):
        with tf.name_scope("dataset"):
            with tf.name_scope("loading"):
                dataset = ThermoDataset(batch_size=batch_size)
                dataset.set_train_test_validation_fraction(train_fraction=0.8, test_fraction=0.2,
                                                           validation_fraction=0.0)

                dataset.load_dataset(root_directory=input_data_path, class_list=thermo_class_list)
                dataset.print_info()

            with tf.name_scope("iterator"):
                train_iterator = dataset.get_train_iterator()
                next_train_batch = train_iterator.get_next()
                test_iterator = dataset.get_test_iterator()
                next_test_batch = test_iterator.get_next()

    with tf.name_scope("placeholders"):
        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [None, 24, 30, 1], name="input_image")
        y = tf.placeholder(tf.int32, [None, dataset.num_classes], name="input_labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_probab")

    # Initialize model
    model = SimpleNet(x=x, num_classes=dataset.num_classes, keep_prob=keep_prob)

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        # Link variable to model output
        logits = model.logits
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y),
                              name="cross_entropy_loss")

    # Add the loss to summary
    tf.summary.scalar('train cross_entropy', loss, collections=["train"])
    tf.summary.scalar('test cross_entropy', loss, collections=["test"])

    # Gradient computation
    with tf.name_scope("gradients"):
        with tf.name_scope("all_gradients"):
            all_gradients = tf.gradients(loss, tf.trainable_variables(), name="all_gradients")
            all_gradients = list(zip(all_gradients, tf.trainable_variables()))

        with tf.name_scope("conv_gradients"):
            gradient_variables = [var for var in tf.trainable_variables() if "conv_" in var.name]
            gradients = tf.gradients(loss, gradient_variables, name="gradients")
            gradients = list(zip(gradients, gradient_variables))
            for grad, var in gradients:
                tf.summary.histogram(var.name + '_gradient', grad, collections=["gradients"])

        with tf.name_scope("full_conn_gradients"):
            gradient_variables = [var for var in tf.trainable_variables() if "full_" in var.name]
            gradients = tf.gradients(loss, gradient_variables, name="gradients")
            gradients = list(zip(gradients, gradient_variables))
            for grad, var in gradients:
                tf.summary.histogram(var.name + '_gradient', grad, collections=["gradients"])

    # Train op
    with tf.name_scope("train"):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, name="optimizer")
        train_op = optimizer.apply_gradients(grads_and_vars=all_gradients, name="apply_gradients")

    # Predict op
    with tf.name_scope("predict"):
        predict_op = tf.argmax(logits, axis=1, name="model_predictions")

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(predict_op, tf.argmax(y, 1), name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy, collections=["train"])
    tf.summary.scalar('accuracy', accuracy, collections=["test"])

    # with tf.name_scope('batch_input'):
    #     tf.summary.image('input images', x, max_outputs=8, collections=["test"])

    # Merge all summaries together
    train_summaries = tf.summary.merge_all(key="train")
    test_summaries = tf.summary.merge_all(key="test")
    gradient_summaries = tf.summary.merge_all(key="gradients")

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(summary_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), summary_path))

        # Loop over number of epochs
        global_step = 0
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch))

            # Initialize iterator with the training and test dataset.
            sess.run(train_iterator.initializer)
            sess.run(test_iterator.initializer)

            confusion_matrix = np.zeros(shape=[3, 3])
            while True:
                # get next batch of data
                try:
                    img_batch, label_batch = sess.run(next_train_batch)
                except tf.errors.OutOfRangeError:
                    print("{} Ended epoch {}".format(datetime.now(), epoch))
                    break

                global_step += 1
                print("{} Global step: {}".format(datetime.now(), global_step))

                # And run the training op
                _, predictions = sess.run([train_op, predict_op], feed_dict={x: img_batch,
                                                                             y: label_batch,
                                                                             keep_prob: keep_probability})

                for l, p in zip(np.argmax(label_batch, axis=1), predictions):
                    confusion_matrix[l, p] += 1

                if global_step % write_train_summaries_every_n_steps == 0:
                    print("{} Writing training summary".format(datetime.now()))
                    train_s, gradient_s = sess.run(
                        [train_summaries, gradient_summaries],
                        feed_dict={x: img_batch, y: label_batch, keep_prob: keep_probability})
                    writer.add_summary(train_s, global_step)

                if global_step % save_model_every_n_steps == 0:
                    print("{} Saving checkpoint of model".format(datetime.now()))

                    # save checkpoint of the model
                    checkpoint_name = os.path.join(checkpoint_path, 'simple_model')
                    save_path = saver.save(sess, checkpoint_name,
                                           global_step=int(global_step / save_model_every_n_steps))

                    print("{} Model checkpoint saved at {}".format(datetime.now(), save_path))

            print("{} Training confusion matrix:\n{}".format(datetime.now(), confusion_matrix))

            print("{} Starting evaluation on test set.".format(datetime.now()))
            # Evaluate on test dataset
            confusion_matrix = np.zeros(shape=[3, 3])
            while True:
                try:
                    img_batch, label_batch = sess.run(next_test_batch)
                except tf.errors.OutOfRangeError:
                    print("{} Test evaluation terminated.".format(datetime.now()))
                    break

                predictions = sess.run(predict_op, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                for l, p in zip(np.argmax(label_batch, axis=1), predictions):
                    confusion_matrix[l, p] += 1

            print("{} Test confusion matrix:\n{}".format(datetime.now(), confusion_matrix))

            if epoch % write_test_summaries_every_n_epochs:
                s = sess.run(test_summaries, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                writer.add_summary(s, global_step)

                with tf.name_scope('image_prediction'):
                    imgs = img_batch[:10]
                    lab = np.argmax(label_batch[0:10, :], axis=1)
                    pred = predictions[0:10]
                    for im, l, p in zip(imgs, lab, pred):
                        image_summary = tf.summary.image("True lab: {}, predicted: {}".format(l, p), np.array([im]))
                        i_s = sess.run(image_summary,
                                       feed_dict={x: img_batch, y: label_batch, keep_prob: keep_probability})
                        writer.add_summary(i_s, global_step)


if __name__ == '__main__':
    main()
