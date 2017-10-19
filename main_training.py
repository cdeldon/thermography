from training.thermo_class import ThermoClass
from training.dataset import ThermoDataset
from training.simple_net import SimpleNet

import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.data import Iterator

from progressbar import Bar, ETA, ProgressBar


def main():
    path = "C:/Users/Carlo/Desktop/Ghidoni"

    working_class = ThermoClass("working", 0)
    broken_class = ThermoClass("broken", 1)
    misdetected_class = ThermoClass("misdetected", 2)
    thermo_class_list = [working_class, broken_class, misdetected_class]

    # Learning params
    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.01

    # Network params
    keep_probability = 0.8

    # Summary params
    write_every_n_steps = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = os.path.join(path, "tensorboard")
    checkpoint_path = os.path.join(path, "checkpoints")

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        with tf.name_scope("dataset"):
            with tf.name_scope("loading"):
                dataset = ThermoDataset(batch_size=batch_size, shuffle=True, buffer_size=1000)
                dataset.load_dataset(root_directory=path, class_list=thermo_class_list)

            with tf.name_scope("iterator"):
                # create an reinitializable iterator given the dataset structure
                iterator = Iterator.from_structure(dataset.data.output_types,
                                                   dataset.data.output_shapes)
                next_batch = iterator.get_next()

                # Ops for initializing the two different iterators
                data_init_op = iterator.make_initializer(dataset.data)

    with tf.name_scope("placeholders"):
        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [batch_size, 24, 30, 1], name="input_image")
        y = tf.placeholder(tf.float32, [batch_size, dataset.num_classes], name="input_labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_probab")

    # Initialize model
    model = SimpleNet(x, keep_prob, dataset.num_classes)

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        # Link variable to model output
        logits = model.y_conv

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y),
                              name="cross_entropy_loss")

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

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
                tf.summary.histogram(var.name + '_gradient', grad)

        with tf.name_scope("full_conn_gradients"):
            gradient_variables = [var for var in tf.trainable_variables() if "full_" in var.name]
            gradients = tf.gradients(loss, gradient_variables, name="gradients")
            gradients = list(zip(gradients, gradient_variables))
            for grad, var in gradients:
                tf.summary.histogram(var.name + '_gradient', grad)

    # Train op
    with tf.name_scope("train"):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, name="optimizer")
        train_op = optimizer.apply_gradients(grads_and_vars=all_gradients, name="apply_gradients")

    # Predict op
    with tf.name_scope("predict"):
        predict_op = tf.argmax(logits, axis=1, name="model_predictions")

    # Add the variables we train to the summary
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(predict_op, tf.argmax(y, 1), name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="sccuracy")

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('batch_input'):
        tf.summary.image('input images', x, max_outputs=8)

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

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        global_step = 0
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            # Initialize iterator with the training dataset.
            sess.run(data_init_op)

            pbar = ProgressBar(widgets=[Bar(), ' ', ETA()], maxval=batches_per_epoch).start()
            for step in range(batches_per_epoch):
                global_step += 1
                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)

                # And run the training op
                _, predictions = sess.run([train_op, predict_op], feed_dict={x: img_batch,
                                                                             y: label_batch,
                                                                             keep_prob: keep_probability})

                if (global_step + 1) % write_every_n_steps == 0:
                    with tf.name_scope('image_prediction'):
                        imgs = img_batch[:10]
                        lab = np.argmax(label_batch[0:10, :], axis=1)
                        pred = predictions[0:10]
                        for im, l, p in zip(imgs, lab, pred):
                            new_summary = tf.summary.image("True lab: {}, predicted: {}".format(l, p), np.array([im]))
                            n_s = sess.run(new_summary, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                            writer.add_summary(n_s, epoch * batches_per_epoch + step)

                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})

                    writer.add_summary(s, epoch * batches_per_epoch + step)
                pbar.update(step + 1)
            pbar.finish()

            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))


if __name__ == '__main__':
    main()
