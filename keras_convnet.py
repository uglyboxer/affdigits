import csv
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import pandas as pd

import theano
print(theano.config.device)

import numpy as np
from affnist_read import loadmat
from tqdm import trange

from img_handler import downsize


# (X_train, y_train), (X_test, y_test) = mnist.load_data()

dataset = loadmat('1.mat')
y_train = dataset['affNISTdata']['label_int']
X_train = dataset['affNISTdata']['image'].transpose()

for i in trange(8):
    dataset1 = loadmat(str(i+1) + '.mat')
    y_train1 = dataset1['affNISTdata']['label_int']
    X_train1 = dataset1['affNISTdata']['image'].transpose()

    X_train = np.vstack((X_train, X_train1))
    y_train = np.hstack((y_train, y_train1))

dataset = loadmat('16.mat')
y_test = dataset['affNISTdata']['label_int']
X_test = dataset['affNISTdata']['image'].transpose()

'''Train a simple convnet on the MNIST dataset.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 40, 40
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

f = open('report.txt', 'w')
f.write('Test score:', score[0])
f.write('Test accuracy:', score[1])
f.close()


# parse Kaggle test set data
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    raw_nums = list(reader)
    test_set = [np.array([int(x) for x in y]) for y in raw_nums[1:]]
testX = []
for i, x in enumerate(test_set):
    testX.append(downsize(x, 28, 40))


# Output Kaggle guess list
testY = model.predict_classes(testX, verbose=2)

pd.DataFrame({"ImageId": list(range(1,len(testY)+1)), "Label": testY}).to_csv('submission.csv', index=False, header=True)
