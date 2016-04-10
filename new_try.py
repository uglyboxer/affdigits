import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.io as spio
from skimage.util import pad

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


test = pd.read_csv("test.csv").values


dataset = loadmat('data3/1.mat')
target = dataset['affNISTdata']['label_int']
train = dataset['affNISTdata']['image'].transpose()

for i in range(14):
    dataset1 = loadmat('data3/' + str(i+2) + '.mat')
    y_train1 = dataset1['affNISTdata']['label_int']
    X_train1 = dataset1['affNISTdata']['image'].transpose()

    train = np.vstack((train, X_train1))
    target = np.hstack((target, y_train1))
    
test = [np.array(x).reshape((28,28)) for x in test]
test = [pad(x, 6, padwithtens).flatten() for x in test]

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 40, 40)).astype(np.float32)
test = np.array(test).reshape((-1, 1, 40, 40)).astype(np.float32)

train = train/255
test = test/255


def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer), 
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 40, 40),
    conv1_num_filters=20,                     
    conv1_filter_size=(5, 5), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=60, 
    conv2_filter_size=(5, 5),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
        
    hidden3_num_units=500,
    output_num_units=10,
    dropout3_p=0.5,
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    max_epochs=n_epochs,
    verbose=1,
    )
    return net1
print(test.shape)
cnn = CNN(10).fit(train,target) # train the CNN model for 15 epochs

pred = cnn.predict(test)
# save results
np.savetxt('submission_cnn.csv', np.c_[range(1, len(test) + 1), pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
