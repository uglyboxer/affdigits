from keras.datasets import mnist

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from affnist_read import loadmat
from tqdm import trange

img_rows = 28
img_cols = 28

b_size = 16

nb_classes = 10

orig_input_dim = img_cols * img_rows

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

# X_train = X_train.reshape(60000, 1600)
# X_test = X_test.reshape(10000, 1600)
X_train = X_train.astype("float64")
X_test = X_test.astype("float64")
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(100, input_dim=1600, init='uniform'))
model.add(Activation('tanh'))
# model.add(Dropout(0.1))
model.add(Dense(100, init='uniform'))
model.add(Activation('tanh'))
# model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.15, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, class_mode='categorical')

model.fit(X_train, Y_train, nb_epoch=10, batch_size=b_size)
score = model.evaluate(X_test, Y_test, batch_size=b_size, show_accuracy=True)
print(model.layers[2].get_weights())
print(score)