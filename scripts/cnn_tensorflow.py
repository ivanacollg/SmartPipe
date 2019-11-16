# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:25:03 2019

@author: DrColula and Rbtote
"""


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib as plt
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255-0.5,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255-0.5)

train_set = train_datagen.flow_from_directory('../images/training',
                                                 target_size = (240, 320), #Tamano de la imagen
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../images/validation',
                                            target_size = (240, 320), #Tamano de la imagen
                                            batch_size = 32,
                                            class_mode = 'binary')
print(train_set) # (60000, 28, 28)
print(train_set) # (60000,)

# Initialising the CNN
classifier = Sequential()

#  Conv-Pool structure 1
classifier.add(Conv2D(4, (3, 3), input_shape = (240, 320,3), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

# Conv-Pool structure 2
classifier.add(Conv2D(8, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

classifier.add(Flatten())
#Full connection
classifier.add(Dense(units = 2400, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'softmax'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.summary()

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.fit(train_set, epochs=5, validation_data=(test_set),verbose=1)


classifier.fit_generator(train_set,
                         steps_per_epoch = 1000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 1000)

classifier.save("Baseline.h5")