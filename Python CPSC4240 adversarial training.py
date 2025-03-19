# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:24:51 2023

@author: colto
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10,mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow import keras
class CIFAR10Model:
    def __init__(self):
        # Load the CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Normalize the pixel values
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        # Convert the labels to one-hot encoded vectors
        self.num_classes = 10
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        # Define the model architecture
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train the model
        batch_size = 32
        epochs = 10
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test, y_test))
        
    def test_model(self, x_test, y_test):
        # Normalize the pixel values
        x_test = x_test / 255.0
        
        # Convert the labels to one-hot encoded vectors
        y_test = to_categorical(y_test, self.num_classes)
        
        # Evaluate the model on the test set
        loss, accuracy = self.model.evaluate(x_test, y_test)
        
        # Print the test set accuracy
        print(f"Test set accuracy: {accuracy}")
        
        # Create a pie chart to visualize the results
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']
        test_predictions = self.model.predict(x_test)
        test_predictions = tf.argmax(test_predictions, axis=1)
        y_test = tf.argmax(y_test, axis=1)
        cm = tf.math.confusion_matrix(y_test, test_predictions)
        cm = cm.numpy()
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(cm[0], labels=class_names, autopct='%1.1f%%')
        ax.axis('equal')
        ax.set_title('Test set accuracy')
        plt.savefig('cifar10_pie_chart.png')
        plt.close()

class MNISTModel:
    def __init__(self):
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the pixel values
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Convert the labels to one-hot encoded vectors
        self.num_classes = 10
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        # Reshape the images to (28, 28, 1) and define the model architecture
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Train the model
        batch_size = 32
        epochs = 10
        self.model.fit(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

    def test_model(self, x_test, y_test):
        # Normalize the pixel values
        x_test = x_test / 255.0
        
        # Convert the labels to one-hot encoded vectors
        y_test = to_categorical(y_test, self.num_classes)
        
        # Evaluate the model on the test set
        loss, accuracy = self.model.evaluate(x_test, y_test)
        
        # Print the test set accuracy
        print(f"Test set accuracy: {accuracy}")
        
        # Create a pie chart to visualize the results
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        test_predictions = self.model.predict(x_test)
        test_predictions = tf.argmax(test_predictions, axis=1)
        y_test = tf.argmax(y_test, axis=1)
        cm = tf.math.confusion_matrix(y_test, test_predictions)
        cm = cm.numpy()
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(cm[0], labels=class_names, autopct='%1.1f%%')
        ax.axis('equal')
        ax.set_title('Test set accuracy')
        plt.savefig('mnist_pie_chart.png')
        plt.close()

# Create an instance of the CIFAR10Model class
model = CIFAR10Model()
mnist_model = MNISTModel()

# Load the CIFAR-10 test set
(x_test, y_test) = cifar10.load_data()[1]
(x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]

# Test the model on the test set
model.test_model(x_test, y_test)
mnist_model.test_model(x_test, y_test)
    