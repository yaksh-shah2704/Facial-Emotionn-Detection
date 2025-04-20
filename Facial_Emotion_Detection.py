# Neural Network
# Epochs = how many times the modal want to go through the images
# batch size = gives the modal the number of images at a time

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.initializers import HeNormal
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt 

import os

num_classes = 5
img_rows, img_cols = 48, 48
batch_size = 8

train_data_dir = r'D:\Advance Python\Emotion_Detection\Emotions\train'
test_data_dir = r'D:\Advance Python\Emotion_Detection\Emotions\test'

train_datagen = ImageDataGenerator(rescale =1./255,
                                   rotation_range =30, shear_range =0.3, zoom_range =0.3, width_shift_range =0.4, height_shift_range =0.4, horizontal_flip =True, vertical_flip =True) 
# zoom = how much prcent we have to zoom the image

# why 255 = all images pixels in rgb form and they have largest number of pixel is 255 and we have to rescale all images into a perticular size and training data size will be reduced(easy to train).

# Do not need to do the same thing as training data because we already trained it and now we have to test the data.
# Whatever it is always rescale the data
test_datagen = ImageDataGenerator(rescale =1./255)

# Here we have categorical data
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                    color_mode = 'grayscale', 
                                                    target_size = (img_rows, img_cols), 
                                                    batch_size = batch_size, 
                                                    class_mode = 'categorical', 
                                                    shuffle = True)

test_generator = test_datagen.flow_from_directory(test_data_dir, 
                                                    color_mode = 'grayscale', 
                                                    target_size = (img_rows, img_cols), 
                                                    batch_size = batch_size, 
                                                    class_mode = 'categorical', 
                                                    shuffle = True)

# CNN Model
# CNN Works in layers it has more than 5-6 layers to generate an output.

model = Sequential()


# Block-1

model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = HeNormal(), input_shape = (img_rows, img_cols, 1)))
model.add(Activation('elu'))        # 'elu' is an activation function just like sigmoid function
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))


# Block-2

model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))        # 'elu' is an activation function just like sigmoid function
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))


# Block-3

model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))        # 'elu' is an activation function just like sigmoid function
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))


# Block-4

model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))        # 'elu' is an activation function just like sigmoid function
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))


# Block-5

model.add(Flatten())
model.add(Dense(64, kernel_initializer = HeNormal()))
model.add(Activation('elu'))        # 'elu' is an activation function just like sigmoid function
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block-6

# model.add(Flatten())
model.add(Dense(64, kernel_initializer = HeNormal()))
model.add(Activation('elu'))        # 'elu' is an activation function just like sigmoid function
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block-7

model.add(Dense(num_classes, kernel_initializer = HeNormal()))
model.add(Activation('softmax'))


print(model.summary())

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor = 'val_loss',
                             mode = 'min',
                             save_best_only = True,
                             verbose = 1
                             )

earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True
                          )

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001
                              )

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(learning_rate=0.001),
              metrics = ['accuracy'])

nb_train_samples = 24176
nb_test_samples = 3006
epochs = 25

earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples//batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = test_generator,
    validation_steps = nb_test_samples//batch_size)


# Accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy")
plt.show()

# Loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss")
plt.show()