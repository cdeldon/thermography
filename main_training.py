from training.thermo_class import ThermoClass
from training.dataset import ThermoDataset
from training.simple_net import SimpleNet

import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.data import Iterator


def main():
    path = "C:/Users/Carlo/Desktop/Ghidoni"

    working_class = ThermoClass("working", 0)
    broken_class = ThermoClass("broken", 1)
    misdetected_class = ThermoClass("misdetected", 2)
    thermo_class_list = [working_class, broken_class, misdetected_class]

    # Learning params
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 24

    # Network params
    dropout_rate = 0.8

    # How often we want to write the tf.summary data to disk
    display_step = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = os.path.join(path, "tensorboard")
    checkpoint_path = os.path.join(path, "checkpoints")

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        dataset = ThermoDataset(batch_size=batch_size, shuffle=True, buffer_size=1000)
        dataset.load_dataset(root_directory=path, class_list=thermo_class_list)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(dataset.data.output_types,
                                           dataset.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    data_init_op = iterator.make_initializer(dataset.data)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 24, 30, 1])
    y = tf.placeholder(tf.float32, [batch_size, dataset.num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = SimpleNet(x, keep_prob, dataset.num_classes)

    # Link variable to model output
    score = model.y_conv

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                      labels=y))

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, tf.trainable_variables())
        gradients = list(zip(gradients, tf.trainable_variables()))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    batches_per_epoch = int(np.floor(dataset.data_size / batch_size))

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            print("Starting epoch")
            sess.run(data_init_op)
            for step in range(batches_per_epoch):
                print("Step {}".format(step))
                img_batch, label_batch = sess.run(next_batch)
                print("Loaded batch")
                print(sess.run(tf.shape(img_batch)))
                print(sess.run(tf.shape(label_batch)))
            print("Epoch ended")

        exit(0)


        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            # Initialize iterator with the training dataset
            sess.run(data_init_op)

            test_acc = 0.
            test_count = 0
            for step in range(batches_per_epoch):

                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)

                # And run the training op
                sess.run(train_op, feed_dict={x: img_batch,
                                              y: label_batch,
                                              keep_prob: dropout_rate})

                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})

                    writer.add_summary(s, epoch * batches_per_epoch + step)
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))


if __name__ == '__main__':
    main()
