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

def dataProc(fpath, correction=0.2):
    lines = []

    with open(fpath + '/driving_log.csv') as labels:
        reader = csv.reader(labels)

        for line in reader:
            lines.append(line)
    
    dirs = [x[0] for x in os.walk(fpath)]
    imgDirs = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), dirs))

    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []

    for imgdir in imgDirs:
        center = []
        left = []
        right = []
        measurements = []

        for line in lines:
            measurements.append(float(line[3]))
            center.append(imgdir + '/' + line[0].strip())
            left.append(imgdir + '/' + line[1].strip())
            right.append(imgdir + '/' + line[2].strip())

        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    imagePaths = []
    measurements = []

    imagePaths.extend(centerTotal)
    imagePaths.extend(leftTotal)
    imagePaths.extend(rightTotal)

    measurements.extend(measurementTotal)
    measurements.extend([x + correction for x in measurementTotal])
    measurements.extend([x - correction for x in measurementTotal])

    return (imagePaths, measurements)

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

