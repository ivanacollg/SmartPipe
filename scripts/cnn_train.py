import numpy as np
import argparse as ap
import matplotlib as plt

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
# Se requieren dos carpetas: training_set y test_set.
# Dentro de cada una, van dos carpetas, una por cada clase.

train_datagen  = ImageDataGenerator(rescale = 1./255)
test_datagen   = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory(path_train,
                                              target_size = (240, 320), #Tamano de la imagen
                                              batch_size = 32,
                                              class_mode = 'binary')

test_set = test_datagen.flow_from_directory(path_test,
                                            target_size = (240, 320),   #Tamano de la imagen
                                            batch_size = 32,
                                            class_mode = 'binary')

print(train_set) # (60000, 28, 28)
print(train_set) # (60000,)

# Initializing the CNN
classifier = Sequential()

# Conv-Pool structure 1
classifier.add(Conv2D(4, (3, 3), input_shape = (240, 320, 3), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis=-1, 
                                  momentum=0.99, 
                                  epsilon=0.001, 
                                  center=True, 
                                  scale=True, 
                                  beta_initializer='zeros', 
                                  gamma_initializer='ones', 
                                  moving_mean_initializer='zeros', 
                                  moving_variance_initializer='ones', 
                                  beta_regularizer=None, 
                                  gamma_regularizer=None, 
                                  beta_constraint=None, 
                                  gamma_constraint=None))

# Conv-Pool structure 2
classifier.add(Conv2D(8, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis=-1, 
                                  momentum=0.99, 
                                  epsilon=0.001, 
                                  center=True, 
                                  scale=True, 
                                  beta_initializer='zeros', 
                                  gamma_initializer='ones', 
                                  moving_mean_initializer='zeros', 
                                  moving_variance_initializer='ones', 
                                  beta_regularizer=None, 
                                  gamma_regularizer=None, 
                                  beta_constraint=None, 
                                  gamma_constraint=None))

classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 2400, activation = 'relu'))
classifier.add(Dense(units = 32,   activation = 'softmax'))
classifier.add(Dense(units = 1,    activation = 'sigmoid'))

# Output summary of the CNN Configuration to the terminal
classifier.summary()

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN
classifier.fit(train_set, epochs=3, validation_data=(test_set))

# Evaluate (validate) the CNN
test_loss, test_acc = classifier.evaluate(test_set, verbose=2)
print('Test Accuracy: %.2f' % (test_acc * 100))

# Save the weights to disk
classifier.save_weights('../weights/smart_weights.h5')

# Save the model as a hdf5 file for future usage
yaml_string = classifier.to_yaml()
with open("../models/nn_model.yaml", "w") as yaml_file:
    yaml_file.write(yaml_string)
