import tensorflow as tf
import numpy as np


class LaNet(tf.Module):

    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        super(LaNet, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_out = n_out
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(n_out)  # logits, no softmax
        ])

    def accuracy(self, logits, labels):
        predicted_classes = tf.argmax(logits, axis=1)
        correct_predictions = tf.equal(predicted_classes, tf.cast(labels, tf.int64))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def train_step(self, x, y):
        """Train one batch with sparse categorical labels"""
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss(logits, y)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value

    def fit(self, x_train, y_train, batch_size=64, epochs=10):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataset:
                batch_x = tf.cast(batch_x, dtype=tf.float32)
                batch_y = tf.cast(batch_y, dtype=tf.int32)
                loss_value = self.train_step(batch_x, batch_y)
                epoch_loss += loss_value.numpy()

            avg_epoch_loss = epoch_loss / len(x_train) * batch_size
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    def evaluate(self, x_data, y_data, batch_size=64):
        accuracy_values = []
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)

        for batch_x, batch_y in dataset:
            batch_x = tf.cast(batch_x, dtype=tf.float32)
            batch_y = tf.cast(batch_y, dtype=tf.int32)
            logits = self.model(batch_x, training=False)
            batch_accuracy = self.accuracy(logits, batch_y)
            accuracy_values.append(batch_accuracy)

        avg_accuracy = tf.reduce_mean(accuracy_values)
        return avg_accuracy.numpy()

    def y_predict(self, x_data, batch_size=64):
        y_pred = []
        for batch_x in tf.data.Dataset.from_tensor_slices(x_data).batch(batch_size):
            batch_x = tf.cast(batch_x, dtype=tf.float32)
            logits = self.model(batch_x, training=False)
            batch_predictions = tf.argmax(logits, axis=1)
            y_pred.extend(batch_predictions.numpy())
        return np.array(y_pred)

    def loss(self, logits, labels):
        """Sparse categorical loss (expects integer labels)"""
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        )


class VGGnet(tf.Module):

    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        super(VGGnet, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_out = n_out
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_out)  # logits
        ])

    def loss(self, logits, labels_onehot):
        """Cross-entropy loss, expects one-hot labels"""
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=logits)
        )

    def accuracy(self, logits, labels):
        y_onehot = tf.one_hot(labels, depth=self.n_out)
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_onehot, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            y_onehot = tf.one_hot(y, depth=self.n_out)
            loss_value = self.loss(logits, y_onehot)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value

    def fit(self, x_train, y_train, batch_size=64, epochs=10):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataset:
                batch_x = tf.cast(batch_x, dtype=tf.float32)
                batch_y = tf.cast(batch_y, dtype=tf.int32)
                loss_value = self.train_step(batch_x, batch_y)
                epoch_loss += loss_value.numpy()

            avg_epoch_loss = epoch_loss / len(x_train) * batch_size
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    def evaluate(self, x_data, y_data, batch_size=64):
        accuracy_values = []
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)

        for batch_x, batch_y in dataset:
            batch_x = tf.cast(batch_x, dtype=tf.float32)
            batch_y = tf.cast(batch_y, dtype=tf.int32)
            logits = self.model(batch_x, training=False)
            batch_accuracy = self.accuracy(logits, batch_y)
            accuracy_values.append(batch_accuracy)

        avg_accuracy = tf.reduce_mean(accuracy_values)
        return avg_accuracy.numpy()

    def y_predict(self, x_data, batch_size=64):
        y_pred = []
        for batch_x in tf.data.Dataset.from_tensor_slices(x_data).batch(batch_size):
            batch_x = tf.cast(batch_x, dtype=tf.float32)
            logits = self.model(batch_x, training=False)
            batch_predictions = tf.argmax(logits, axis=1)
            y_pred.extend(batch_predictions.numpy())
        return np.array(y_pred)
