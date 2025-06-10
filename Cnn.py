import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load and filter CIFAR-10 for 4 classes (Car, Cat, Dog, Person -> airplane, cat, dog, truck as proxy)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Filter to 4 classes: airplane(0), cat(3), dog(5), truck(9)
classes_to_use = [0, 3, 5, 9]
filter_train = np.isin(y_train, classes_to_use).flatten()
filter_test = np.isin(y_test, classes_to_use).flatten()

x_train, y_train = x_train[filter_train], y_train[filter_train]
x_test, y_test = x_test[filter_test], y_test[filter_test]

# Map labels to [0, 1, 2, 3]
label_map = {v: i for i, v in enumerate(classes_to_use)}
y_train = np.vectorize(label_map.get)(y_train)
y_test = np.vectorize(label_map.get)(y_test)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate
y_pred = np.argmax(model.predict(x_test), axis=1)

# Classification Report & Confusion Matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
