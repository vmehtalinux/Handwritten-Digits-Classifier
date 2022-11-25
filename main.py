# Importing all of the needed modules
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import cv2

# Loading the data from mnist handwritten digits database
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Shows the array values for the first image
# print(train_images[0])

# Graphs the first image in the data set
"""
plt.imshow(train_images[0])
plt.show()
"""

# Graphs the first 100 images in the training set
"""
plt.figure(figsize=(10,10))
for i in range(100):
  plt.subplot(10,10,i+1)
  plt.imshow(train_images[i])
plt.show()
"""

# Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

# Compiling and optimizing the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fitting and training the model
model.fit(train_images, train_labels, epochs=10)
test_loss,test_acc=model.evaluate(test_images, test_labels)

# Predicting all of the images in the test dataset
predictions=model.predict(test_images)

# Adding up all of our data values
values=[1.1812429e-33, 1.1375997e-17, 6.9389866e-13, 6.2058149e-12, 5.7243868e-23,
 3.9824978e-21, 0.0000000e+00, 1.0000000e+00, 4.0298552e-22, 8.0003461e-19]
print(sum(values))
