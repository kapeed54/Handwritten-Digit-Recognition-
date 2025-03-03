#1. loading and exploring the data:
#We'll use the MNIST dataset, which contains 70,000 grayscale 
# images (28x28 pixels) of handwritten digits (0-9).

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#a. load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#b. print dataset shape
print(f"Training data: {x_train.shape}, Labels: {y_train.shape}")
print(f"Testing data: {x_test.shape}, Labels: {y_test.shape}")

#b. visualize some sample images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    plt.show()

# x_train contains images of digits (28x28 grayscale pixels).
# y_train contains labels (actual numbers: 0,1,2,…,9).
# We visualize the dataset to understand what AI will learn from.


#2. Preprocessing the data:

# a. Normalize images (convert pixel values from [0, 255] to [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

#b. Reshape data to fit into a neural network
x_train = x_train.reshape(-1, 28*28) #flatten 28x28 images to 1D (784)
x_test = x_test.reshape(-1, 28*28)

print(f"New Training shape: {x_train.shape}") #shape: (60000, 784)

# Normalization: Makes training faster & stable.
# Flattening: Converts 2D images into 1D vectors (needed for neural networks).

#3. Building a neural network model:
# a. Define the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)), #hidden layer
    tf.keras.layers.Dense(10, activation='softmax') #output layer
])

#b. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#c. Print model summary
model.summary() 

# Dense Layers: Fully connected layers where each neuron is linked to all neurons in the next layer.
# ReLU Activation: Helps the network learn non-linear patterns.
# Softmax Activation: Converts output into probabilities (sum = 1).

#4. Training the model:
# a. Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Epochs: One full pass through the training data.
# Backpropagation: AI updates weights to reduce errors.
# Adam Optimizer: Adjusts learning rates dynamically.

#5. Evaluating the model performance:
# a. Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Accuracy tells us how many test images were classified correctly.

#6. Making predictions:
# a. Make predictions on test images
predictions = model.predict(x_test)

#b. plot some predictions
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

#np.argmax(predictions[i]) picks the digit with the highest probability.

#7. Improve and experiment:
# a. Increase the number of epochs for better accuracy.
# b. Add more hidden layers to learn complex patterns.