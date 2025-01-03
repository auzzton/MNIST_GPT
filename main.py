import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Load the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize the data
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# # Reshape the data for Conv2D
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# # Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )
# datagen.fit(x_train)

# # Build the model
# model = tf.keras.models.Sequential()

# # Add Convolutional layers
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# # Flatten and add Dense layers
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))  # Dropout to prevent overfitting
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model with augmented data
# model.fit(datagen.flow(x_train, y_train), epochs=3)

# # Saving the model 
# model.save('MNIST_Model.keras')

# Loading the saved model
model = tf.keras.models.load_model('MNIST_Model.keras')

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

# Print the results
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Predict on external images
image_number = 1
while os.path.isfile(f"images/digit{image_number}.png"):
    try:
        # Load the image in grayscale mode
        img = cv2.imread(f"images/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)

        # Check if the image was correctly loaded
        if img is None:
            print(f"Image digit{image_number}.png not found or couldn't be opened.")
            break

        # Resize and normalize the image to 28x28 as required by the model
        img = cv2.resize(img, (28, 28))  # Ensure the image is 28x28 pixels
        img = np.invert(img)  # Invert colors (black to white, white to black)
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for Conv2D
        img = np.array([img])  # Add batch dimension to match model input shape (1, 28, 28, 1)

        # Predict the digit
        prediction = model.predict(img)
        print(f"This digit could be a {np.argmax(prediction)}")

        # Display the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except Exception as e:
        print(f"Error processing digit{image_number}.png: {e}")
    finally:
        image_number += 1
