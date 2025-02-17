import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

y_train = df_train['label'].values
x_train = df_train.drop(columns='label').values

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = keras.utils.to_categorical(y_train, 10)


model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train_cat, epochs=5, validation_split=0.2)

model.save("mnist_model.h5")

print("Model training completed and saved as mnist_model.h5")
