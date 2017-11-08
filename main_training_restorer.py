import os
import random

import cv2
import numpy as np
import tensorflow as tf

from thermography.classification.models import ThermoNet

output_path = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/output"
checkpoint_path = os.path.join(output_path, "checkpoints")
input_folder = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/dataset/Ghidoni/0-1000"
num_images = 300

if __name__ == '__main__':
    image_shape = np.array([96, 120, 1])
    num_classes = 3

    with tf.name_scope("placeholders"):
        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [None, *image_shape], name="input_image")

    model = ThermoNet(x=x, image_shape=image_shape, num_classes=num_classes, keep_prob=1.0)

    with tf.name_scope("predict"):
        predict_op = tf.argmax(model.logits, axis=1, name="model_predictions")
        probabilities = tf.nn.softmax(model.logits)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    class_name = {0: "working", 1: "broken", 2: "misdetected"}

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print("Model restored.")

        input_images = []
        print("Loading images..")
        image_per_class = num_images / 3
        for class_type in os.listdir(input_folder):
            true_label = class_type
            folder_path = os.path.join(input_folder, class_type)
            image_count = 0
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                input_images.append(
                    {"image": cv2.imread(img_path, cv2.IMREAD_COLOR), "true_label": true_label, "file_name": img_path})
                image_count += 1
                if image_count > image_per_class:
                    break

        random.shuffle(input_images)
        print("{} images laoded!".format(len(input_images)))

        for input_image in input_images:
            img = input_image["image"]
            true_label = input_image["true_label"]
            image_name = input_image["file_name"]

            resized_img = cv2.resize(img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
            normalized_img = cv2.normalize(resized_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            class_probabilities, predicted_label = sess.run([probabilities, predict_op],
                                                            feed_dict={x: [normalized_img]})

            predicted_correcly = class_name[predicted_label[0]] == true_label
            if predicted_correcly:
                font_color = (40, 200, 40)
            else:
                font_color = (0, 0, 255)
            font_scale = 1.0
            thickness = 2
            cv2.putText(img, "True lab: {}".format(true_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        font_color, thickness)
            cv2.putText(img, "Predicted: {}".format(class_name[predicted_label[0]]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, font_color, thickness)
            np.set_printoptions(precision=3, suppress=True)
            cv2.putText(img, "Logits: {}".format(class_probabilities[0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, font_color, thickness)
            print("Image {}".format(image_name))
            cv2.imshow("Module", img)

            cv2.waitKey(700)
            if (true_label == "broken" and predicted_label[0]!=1) or (true_label!="broken" and predicted_label==1):
                cv2.waitKey()
