import csv, json
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import pandas as pd

from skimage.util import pad

import theano
print(theano.config.device)

import numpy as np
from affnist_read import loadmat
from tqdm import trange, tqdm

from img_handler import downsize

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 40, 40
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

img_cols, img_rows = 40, 40


def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def zeroStack(vector):
    top = np.zeros((3, 28))
    bottom = np.zeros((9, 28))
    left = np.zeros((40, 8))
    right = np.zeros((40, 4))
    vector = np.vstack((top, vector))
    vector = np.vstack((vector, bottom))
    vector = np.hstack((left, vector))
    vector = np.hstack((vector, right))
    return vector


model = Sequential()

model.add(Convolution2D(20, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(60, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(Convolution2D(120, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# EarlyStopping(monitor='val_loss')

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.load_weights('my_model_weights1.h5')

# parse Kaggle test set data
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    raw_nums = list(reader)
    test_set = [np.array([x for x in y]).astype('float32') for y in raw_nums[1:]]

# testX = downsize(test_set[0], 28, 40).flatten()
testX = test_set[0].reshape(28, 28)
# testX = [pad(testX, 6, padwithtens).flatten()]
testX = [zeroStack(testX).flatten()]
for x in tqdm(test_set[1:]):
    # y = downsize(x, 28, 40).flatten()
    y = x.reshape(28, 28)
    # y = pad(y, 6, padwithtens).flatten() 
    y = zeroStack(y).flatten()
    testX.append(y)
length = len(testX)
testX = np.array(testX)
testX = testX.reshape(length, 1, img_rows, img_cols)
testX /= 255

# Output Kaggle guess list
testY = model.predict_classes(testX, verbose=2)

pd.DataFrame({"ImageId": list(range(1,len(testY)+1)), "Label": testY}).to_csv('submission.csv', index=False, header=True)
