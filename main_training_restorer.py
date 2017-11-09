import os
import random

import cv2
import numpy as np

from thermography.classification import Inference
from thermography.classification.models import ThermoNet3x3
from thermography.settings import get_resources_dir
from thermography.utils import scale_image

input_folder = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/padded_dataset/Ghidoni1/0-1000"
num_images = 300

if __name__ == '__main__':

    image_shape = np.array([96, 120, 1])
    num_classes = 3
    checkpoint_dir = os.path.join(get_resources_dir(), "weights")
    inference = Inference(checkpoint_dir=checkpoint_dir, model_class=ThermoNet3x3,
                          image_shape=image_shape, num_classes=num_classes)

    class_name = {0: "working", 1: "broken", 2: "misdetected"}

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

        resized_image = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

        probabilities = inference.classify([img])[0]
        predicted_class = np.argmax(probabilities)
        predicted_correcly = (class_name[predicted_class] == true_label)
        print(probabilities, predicted_class, predicted_correcly)
        if predicted_correcly:
            font_color = (40, 200, 40)
        else:
            font_color = (0, 0, 255)
        font_scale = 1.0
        thickness = 2
        cv2.putText(resized_image, "True lab: {}".format(true_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    font_color, thickness)
        cv2.putText(resized_image, "Predicted: {}".format(class_name[predicted_class]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_color, thickness)
        np.set_printoptions(precision=3, suppress=True)
        cv2.putText(resized_image, "Logits: {}".format(probabilities), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_color, thickness)
        print("Image {}".format(image_name))
        cv2.imshow("Module", resized_image)

        cv2.waitKey(0 if not predicted_correcly else 700)
