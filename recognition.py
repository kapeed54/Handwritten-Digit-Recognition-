#1. loading and exploring the data:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#a. load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print dataset shape
print(f"Training data: {x_tran.shape}, Labels: {y_train.shape}")
print(f"Testing data: {x_test.shape}, Labels: {y_test.shape}")

#b. visualize some sample images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    plt.show()

#2. Preprocessing the data:
# a. Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

#b. Reshape the data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Training data: {x_train.shape}")