new_try.py will run a CNN on matlab file of affine transformed MNIST.

Currently it points to .mat files in a folder called data3/ that is on the same dir level as the main py file.

There are 32 .mat files.  Currently only first 15 are used for training as that is all that will fit in common memory setups.

All input vectors are stacked into a 4d numpy matrix for training and testing.  The dimensions of which are:
(# of vectors, channels, height, width) where channels are 1 for b/w and 3 for rgb

The test set is the Kaggle MNIST and it outputs the appropriate csv format for submission to the competition.