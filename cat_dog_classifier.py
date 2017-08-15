import keras
import numpy as numpy
from parser import load_data

#collect data
training_data = load_data('data/training')
validation_data = load_data('data/validation')

#build model

model = Sequential()

model.add(Convolution2d(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2d(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2d(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('Sigmoid'))

model.compile(loss='binary_crossentropy',
				optimizer='rmsprop', #gradient descent
				metrics=['accuracy'])

model.fit_generator(
	training_data,
	samples_per_epoch=2048,
	nb_epoch=30,
	validation_data=validation_data,
	nb_val_samples=832)

model.save_weights('model/simple_CNN.h5')