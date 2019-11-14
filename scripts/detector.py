import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib as plt

path_train= '../images/train'
path_test = '../images/test'
#Data augmentation y subir dataset
#Se requieren dos carpetas: training_set y test_set. Dentro de cada una, van dos carpetas, una por cada clase.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory(path_train,
                                                 target_size = (240, 320), #Tamano de la imagen
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(path_test,
                                            target_size = (240, 320), #Tamano de la imagen
                                            batch_size = 32,
                                            class_mode = 'binary')
print(train_set) # (60000, 28, 28)
print(train_set) # (60000,)

'''
# Reshape the images.
train_set = np.expand_dims(train_set, axis=3)
test_set= np.expand_dims(test_set, axis=3)

print(train_set.shape) # (60000, 28, 28, 1)
print(test_set.shape)  # (10000, 28, 28, 1)
'''


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


classifier.fit(train_set, epochs=3, validation_data=(test_set))

'''
plt.plot(classifier.history['accuracy'], label='accuracy')
plt.plot(classifier.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
'''
# Save the model to disk.
model.save_weights('../weights/smart_weights.h5')

test_loss, test_acc = classifier.evaluate(test_set, verbose=2)
print('Test Accuracy: %.2f' % (test_acc*100))
