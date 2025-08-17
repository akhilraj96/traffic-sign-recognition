import os
import sys
import numpy as np
import random
import cv2
import skimage.morphology as morp

from skimage.filters import rank
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_data, save_data

# from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('artifacts', 'train.p')
    test_data_path: str = os.path.join('artifacts', 'test.p')
    

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()

  def augment_brightness_camera_images(self, image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

  def transform_image(self, img, ang_range, shear_range, trans_range, brightness=0):
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    if brightness == 1:
      img = self.augment_brightness_camera_images(img)

    return img
  def gray_scale(self, image):
    """
    Convert images to gray scale.
      Parameters:
        image: np.array
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  def local_histo_equalize(self, image):
    """
    Apply local histogram equalization to grayscale images.
      Parameters:
        image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, kernel)
    return img_local

  def image_normalize(self, image):
    """
    Normalize images to [0, 1] scale.
      Parameters:
        image: np.array
    """
    image = np.divide(image, 255)
    return image
  def preprocess(self, data):
    """
    Applying the preprocessing steps to the input data.
      Parameters:
        data: np.array
    """
    gray_images = list(map(self.gray_scale, data))
    equalized_images = list(map(self.local_histo_equalize, gray_images))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
      normalized_images[i] = self.image_normalize(img)
    normalized_images = normalized_images[..., None]
    return normalized_images

  def initiate_data_transformation(self, data_path):
    logging.info("Entered the Data Transformation method")

    try:
      X_data, y_data = load_data(data_path)

      # Number of data examples
      n_train = X_data.shape[0]

      # shape of an traffic sign image
      image_shape = X_data[0].shape

      # unique classes/labels in the dataset
      n_classes = len(np.unique(y_data))

      logging.info("Number of training examples : " + str(n_train))
      logging.info("Image data shape :" + str(image_shape))
      logging.info("Number of classes :" + str(n_classes))

      _classes, counts = np.unique(y_data, return_counts=True)

      for _class, count in zip(_classes, counts):
        new_images = []
        new_classes = []

        if count < 1000:
          y_data_length = y_data.shape[0]
          index = 0

          for i in range(0, 1000-count):
            while y_data[index] != _class: index = random.randint(0, y_data_length-1)
            new_images.append(self.transform_image(X_data[index], 10, 5, 5, brightness=1))
            new_classes.append(_class)
          X_data = np.concatenate((X_data, np.array(new_images)))
          y_data = np.concatenate((y_data, np.array(new_classes)))

      logging.info("Number of training examples after augmenting : " + str(X_data.shape[0]))

      X_data = self.preprocess(X_data)

      X_data, y_data = shuffle(X_data, y_data)

      X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2,random_state=42)

      save_data(X_train, y_train, DataTransformationConfig.train_data_path)
      save_data(X_test, y_test, DataTransformationConfig.test_data_path)

      logging.info("<><><><><><><><><><><><><><><>")

      return (
        DataTransformationConfig.train_data_path,
        DataTransformationConfig.test_data_path)

    except Exception as e:
      raise CustomException(e, sys)


if __name__ == "__main__":
  pass
