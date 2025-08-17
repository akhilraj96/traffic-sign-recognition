import os
import sys
import pandas as pd
from src.exception import CustomException

import csv
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class PredictPipeline:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        return model

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (32, 32))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        gray_images = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        image = np.divide(gray_images, 255)
        return image

    def predict(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(np.array([preprocessed_image]))
        predicted_class = np.argmax(prediction)

        return predicted_class


# Example usage:
if __name__ == "__main__":
    model_path = 'artifacts/Saved_Models/VGGNet'
    image_path = "data/test_images/001.jpg"

    pipeline = PredictPipeline(model_path)
    predicted_class = pipeline.predict(image_path)

    signs = []
    with open('data/signnames.csv', 'r') as csvfile:
        signnames = csv.reader(csvfile, delimiter=',')
        next(signnames, None)
        for row in signnames:
            signs.append(row[1])
        csvfile.close()

    print("Predicted class: ", signs[predicted_class])
