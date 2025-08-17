import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from sklearn.utils import shuffle

from src.components.model import LaNet, VGGnet
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import load_data, load_data_2


class ModelTrainer:
  def __init__(self, batch_size=64, epochs=30, learning_rate=0.001):
    self.batch_size = batch_size
    self.epochs = epochs
    self.learning_rate = learning_rate

  def train(self, train_path, test_path):
    logging.info("Entered the Model Trainer method")
    X_train, y_train = load_data_2(train_path)
    X_test, y_test = load_data_2(test_path)

    n_classes = len(np.unique(y_train))

    # Validation set preprocessing
    X_test_preprocessed = X_test
    DIR = 'artifacts/Saved_Models'

    # Create an instance of the LeNet class
    LeNet_Model = LaNet(n_out=n_classes)
    model_name = "LeNet.h5"

    # Create an optimizer
    optimizer = tf.optimizers.Adam(learning_rate=LeNet_Model.learning_rate)

    logging.info("Starting LeNet Training")
    print()
    print("LeNet Training...")
    print()

    # Training loop
    for epoch in range(self.epochs):
      X_train, y_train = shuffle(X_train, y_train)
      num_batches = len(y_train) // self.batch_size

      for batch_num in range(num_batches):
        start_idx = batch_num * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_x, batch_y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]

        with tf.GradientTape() as tape:
          logits = LeNet_Model.model(batch_x, training=True)
          loss_value = LeNet_Model.loss(logits, batch_y)

        gradients = tape.gradient(loss_value, LeNet_Model.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, LeNet_Model.model.trainable_variables))


      validation_accuracy = LeNet_Model.evaluate(X_test_preprocessed, y_test)
      print("Epoch {} : Validation Accuracy = {:.3f}%".format(epoch + 1, (validation_accuracy * 100)))

    # Save the model
    LeNet_Model.model.save(os.path.join(DIR, model_name))
    logging.info("Model saved in " + str(os.path.join(DIR, model_name)))

    VGGNet_Model = VGGnet(n_out=n_classes)
    model_name = "VGGNet.h5"

    # Initialize optimizer
    optimizer = tf.optimizers.Adam(learning_rate=VGGNet_Model.learning_rate)

    logging.info("Starting VGGNet Training")
    print()
    print("VGGNet Training...")
    print()
    for i in range(self.epochs):
      X_train, y_train = shuffle(X_train, y_train)
      num_batches = len(y_train) // self.batch_size

      # Initialize variables to accumulate losses and accuracies
      total_loss = 0.0
      total_accuracy = 0.0

      for batch_num in range(num_batches):
        start_idx = batch_num * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_x, batch_y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]

        with tf.GradientTape() as tape:
          # Perform a training step and accumulate loss
          loss_value = VGGNet_Model.train_step(batch_x, batch_y)
          total_loss += loss_value

        gradients = tape.gradient(loss_value, VGGNet_Model.model.trainable_variables)
        gradients = [grad if grad is not None else tf.zeros_like(
            var) for grad, var in zip(gradients, VGGNet_Model.model.trainable_variables)]

        optimizer.apply_gradients(zip(gradients, VGGNet_Model.model.trainable_variables))

        # Compute batch accuracy
        logits = VGGNet_Model.model(batch_x)
        batch_accuracy = VGGNet_Model.accuracy(logits, batch_y)
        total_accuracy += batch_accuracy

      # Calculate average loss and accuracy for the epoch
      avg_epoch_loss = total_loss / num_batches
      avg_epoch_accuracy = total_accuracy / num_batches

      # Evaluate the model on the validation set
      validation_accuracy = VGGNet_Model.evaluate(X_test_preprocessed, y_test)

      print("EPOCH {} : Loss = {:.4f}, Training Accuracy = {:.3f}%, Validation Accuracy = {:.3f}%".format(
          i + 1, avg_epoch_loss, avg_epoch_accuracy * 100, validation_accuracy * 100))

    # Save the model using TensorFlow 2.x methods
    VGGNet_Model.model.save(os.path.join(DIR, model_name))
    logging.info("Model saved in " + str(os.path.join(DIR, model_name)))

    return validation_accuracy


if __name__ == "__main__":
    pass
