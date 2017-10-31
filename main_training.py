import os
import timeit
from datetime import datetime

import numpy as np
import tensorflow as tf

from training.dataset import ThermoDataset
from training.models import RGBNet, ComplexNet
from training.thermo_class import ThermoClass


def main():
    dataset_path = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/dataset"
    recording_path_list = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    input_data_path = []
    for g in recording_path_list:
        input_data_path.extend([os.path.join(g, f) for f in os.listdir(g)])

    print("Input data path:")
    for path_index, path in enumerate(input_data_path):
        print("  ({}) {}".format(path_index, path))

    print()
    output_data_path = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/output"
    print("Output data path:\n      {}".format(output_data_path))
    print()

    working_class = ThermoClass("working", 0)
    broken_class = ThermoClass("broken", 1)
    misdetected_class = ThermoClass("misdetected", 2)
    thermo_class_list = [working_class, broken_class, misdetected_class]

    load_all_data = True

    # Learning params
    num_epochs = 100000
    batch_size = 128
    learning_rate = 0.0002

    # Network params
    image_shape = np.array([96, 120, 1])
    keep_probability = 0.5

    # Summary params
    write_train_summaries_every_n_steps = 100
    write_test_summaries_every_n_epochs = 20
    save_model_every_n_epochs = 100

    # Path for tf.summary.FileWriter and to store model checkpoints
    summary_path = os.path.join(output_data_path, "tensorboard")
    checkpoint_path = os.path.join(output_data_path, "checkpoints")

    # Place data loading and preprocessing on the cpu.
    with tf.device('/cpu:0'):
        with tf.name_scope("dataset"):
            with tf.name_scope("loading"):
                dataset = ThermoDataset(batch_size=batch_size, balance_data=True, img_shape=image_shape)
                dataset.set_train_test_validation_fraction(train_fraction=0.8, test_fraction=0.2,
                                                           validation_fraction=0.0)

                dataset.load_dataset(root_directory_list=input_data_path, class_list=thermo_class_list,
                                     load_all_data=load_all_data)
                dataset.print_info()

            with tf.name_scope("iterator"):
                train_iterator = dataset.get_train_iterator()
                next_train_batch = train_iterator.get_next()
                test_iterator = dataset.get_test_iterator()
                next_test_batch = test_iterator.get_next()

    with tf.name_scope("placeholders"):
        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [None, *image_shape], name="input_image")
        y = tf.placeholder(tf.int32, [None, dataset.num_classes], name="input_labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_probab")

    # Initialize model
    model = ComplexNet(x=x, image_shape=image_shape, num_classes=dataset.num_classes, keep_prob=keep_prob)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
        train_op = optimizer.minimize(loss)

    # Predict op
    with tf.name_scope("predict"):
        predict_op = tf.argmax(logits, axis=1, name="model_predictions")

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(predict_op, tf.argmax(y, 1), name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Add the accuracy to the summary
    tf.summary.scalar('train accuracy', accuracy, collections=["train"])
    tf.summary.scalar('test accuracy', accuracy, collections=["test"])

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

        # Add the model graph to TensorBoard only if we did not load the entire dataset!
        if not load_all_data:
            writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir={}".format(datetime.now(), summary_path))

        train_steps_per_epoch = int(np.ceil(dataset.train_size / dataset.batch_size))
        print("{} Number of training steps per epoch: {}".format(datetime.now(), train_steps_per_epoch))
        test_steps_per_epoch = int(np.ceil(dataset.test_size / dataset.batch_size))
        print("{} Number of test steps per epoch: {}".format(datetime.now(), test_steps_per_epoch))
        print()

        # Loop over number of epochs
        global_step = 0
        for epoch in range(num_epochs):

            print("=======================================================")
            print("{} Starting epoch number: {}".format(datetime.now(), epoch))

            # Initialize iterator with the training and test dataset.
            sess.run(train_iterator.initializer)
            sess.run(test_iterator.initializer)

            all_train_predictions = []
            all_train_labels = []
            train_epoch_step = 0
            while True:
                step_start_time = timeit.default_timer()

                # get next batch of data
                try:
                    img_batch, label_batch = sess.run(next_train_batch)
                except tf.errors.OutOfRangeError:
                    print("{} Ended training epoch number {}".format(datetime.now(), epoch))
                    break

                # And run the training op
                _, predictions = sess.run([train_op, predict_op], feed_dict={x: img_batch,
                                                                             y: label_batch,
                                                                             keep_prob: keep_probability})
                all_train_predictions.extend(predictions)
                all_train_labels.extend(np.argmax(label_batch, axis=1))

                if global_step % write_train_summaries_every_n_steps == 0:
                    print("{} Writing training summary".format(datetime.now()))
                    train_s, gradient_s = sess.run(
                        [train_summaries, gradient_summaries],
                        feed_dict={x: img_batch, y: label_batch, keep_prob: keep_probability})
                    writer.add_summary(train_s, global_step)

                step_end_time = timeit.default_timer()

                global_step += 1
                train_epoch_step += 1
                print("{} Global step {}, Epoch: {}, Epoch step {}/{}, ETA: {:.3g} s."
                      .format(datetime.now(), global_step, epoch, train_epoch_step, train_steps_per_epoch,
                              step_end_time - step_start_time))

            cm = tf.confusion_matrix(labels=all_train_labels, predictions=all_train_predictions,
                                     num_classes=dataset.num_classes).eval()
            print("{} Training confusion matrix:\n{}".format(datetime.now(), cm))

            print("-------------------------------------------------------")
            print("{} Starting evaluation on test set.".format(datetime.now()))
            # Evaluate on test dataset
            all_test_predictions = []
            all_test_labels = []
            test_summaries_written = False
            test_epoch_steps = 0
            wrongly_classified = []
            while True:
                step_start_time = timeit.default_timer()
                try:
                    img_batch, label_batch = sess.run(next_test_batch)
                except tf.errors.OutOfRangeError:
                    print("{} Test evaluation terminated.".format(datetime.now()))
                    break

                predictions, predicted_correctly = sess.run([predict_op, correct_pred],
                                                            feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                all_test_predictions.extend(predictions)
                all_test_labels.extend(np.argmax(label_batch, axis=1))

                for img, p, l in zip(img_batch[~predicted_correctly], predictions[~predicted_correctly],
                                     np.argmax(label_batch[~predicted_correctly, :], axis=1)):
                    wrongly_classified.append({"img": img, "prediction": p, "label": l})

                step_end_time = timeit.default_timer()
                test_epoch_steps += 1
                print("{} Epoch: {}, Test epoch step {}/{}, ETA: {:.3g} s."
                      .format(datetime.now(), epoch, test_epoch_steps, test_steps_per_epoch,
                              step_end_time - step_start_time))

                if not test_summaries_written and epoch % write_test_summaries_every_n_epochs == 0:
                    print("{} Writing test summary".format(datetime.now()))
                    test_summaries_written = True
                    s = sess.run(test_summaries, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                    writer.add_summary(s, global_step)

            cm = tf.confusion_matrix(labels=all_test_labels, predictions=all_test_predictions,
                                     num_classes=dataset.num_classes).eval()
            print("{} Test confusion matrix:\n{}".format(datetime.now(), cm))

            if epoch % write_test_summaries_every_n_epochs == 0:
                with tf.name_scope('image_prediction'):
                    if len(wrongly_classified) > 10:
                        wrongly_classified = wrongly_classified[0:10]
                    for wrong in wrongly_classified:
                        image_summary = tf.summary.image(
                            "True lab: {}, predicted: {}".format(wrong["label"], wrong["prediction"]),
                            np.array([wrong["img"]]))
                        i_s = sess.run(image_summary)
                        writer.add_summary(i_s, global_step)

            if epoch % save_model_every_n_epochs == 0:
                print("{} Saving checkpoint of model".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, model.name)
                save_path = saver.save(sess, checkpoint_name, global_step=epoch)

                print("{} Model checkpoint saved at {}".format(datetime.now(), save_path))


if __name__ == '__main__':
    main()
