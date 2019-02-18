import csv
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential

def datasetGen():
    with open('simdata/driving_log.csv') as log_file:
        log_reader = csv.DictReader(log_file)
        X = []
        y = []
        steering_offset = 0.4

        for row in log_reader:
            centerImage = mpimg.imread(row['center'].strip().replace('/home/era/Projects/Work/simdata', 'simdata'))
            flippedCenterImage = np.fliplr(centerImage)
            centerSteering = float(row['steering'])

            leftImage = mpimg.imread(row['left'].strip().replace('/home/era/Projects/Work/simdata', 'simdata'))
            flippedLeftImage = np.fliplr(leftImage)
            leftSteering = centerSteering + steering_offset

            rightImage = mpimg.imread(row['right'].strip().replace('/home/era/Projects/Work/simdata', 'simdata'))
            flippedRightImage = np.fliplr(rightImage)
            rightSteering = centerSteering - steering_offset
            
            X.append(centerImage)
            X.append(flippedCenterImage)
            X.append(leftImage)
            X.append(flippedLeftImage)
            X.append(rightImage)
            X.append(flippedRightImage)
            
            y.append(centerSteering)
            y.append(-centerSteering)
            y.append(leftSteering)
            y.append(-leftSteering)
            y.append(rightSteering)
            y.append(-rightSteering)

    X = np.array(X)
    y = np.array(y)
    
    return X, y

def model():
    model = Sequential()

    model.add(layers.Conv2D(16, 
                            kernel_size=(5, 5), 
                            strides=(2, 2), 
                            activation='relu', 
                            input_shape=(160, 320, 3), 
                            padding='same'))

    model.add(layers.Conv2D(32, 
                            kernel_size=(5, 5), 
                            strides=(2, 2), 
                            activation='relu', 
                            padding='valid'))

    model.add(layers.AveragePooling2D(pool_size=(2, 2), 
                                      strides=(1, 1), 
                                      padding='valid'))

    model.add(layers.Conv2D(64, 
                            kernel_size=(5, 5), 
                            strides=(2, 2), 
                            activation='relu', 
                            padding='valid'))

    model.add(layers.Conv2D(64, 
                            kernel_size=(3, 3), 
                            strides=(2, 2), 
                            activation='relu', 
                            padding='valid'))

    model.add(layers.AveragePooling2D(pool_size=(2, 2), 
                                      strides=(1, 1), 
                                      padding='valid'))

    model.add(layers.Conv2D(128, 
                            kernel_size=(3, 3), 
                            strides=(1, 1), 
                            activation='relu', 
                            padding='valid'))

    model.add(layers.Conv2D(128, 
                            kernel_size=(3, 3), 
                            strides=(1, 1), 
                            activation='relu', 
                            padding='valid'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='linear'))

    model.add(layers.Dense(512, activation='linear'))

    model.add(layers.Dense(64, activation='linear'))

    model.add(layers.Dense(8, activation='linear'))

    model.add(layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

    return model

X, y = datasetGen()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42) 

model = model()
model.summary()

model.fit(X_train, y_train, 
          epochs=8, 
          batch_size=512,
          validation_data=(X_valid, y_valid))

model.save('model.h5')