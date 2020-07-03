import os
import tensorflow as tf
import numpy as np
import pandas as pd

SLICE_INDEX = 100

train_data = pd.read_csv('./data/train.csv', header=0)

y_train_data = np.asarray(train_data['label'])[:SLICE_INDEX]
x_train_data = np.asarray(train_data.drop('label', axis=1))[:SLICE_INDEX]

x_train_data = x_train_data / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28,28,1), input_shape=x_train_data[0].shape),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(x_train_data, y_train_data, epochs=1)

test_data = np.asarray(pd.read_csv('./data/test.csv', header=0))[:SLICE_INDEX]
test_data = [v.reshape(1,784) for v in test_data]
predictions = [model.predict(v) for v in test_data]
formatted_prediction = [np.argmax(v) for v in predictions] 

dataset = pd.DataFrame({'ImageID': np.arange(1, len(formatted_prediction) + 1), 'Label': formatted_prediction})
dataset.to_csv('./prediction/results.csv', index=False)
