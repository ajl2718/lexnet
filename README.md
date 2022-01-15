# lexnet: learning neural nets from basics

**Description**

Coding up a neural network library from scratch in order to learn the various aspects of building, training neural networks.

Currently:
- Feedforward networks with arbitrary numbers of layers and neurons in each layer
- Basic Stochastic Gradient
- Backpropagation (have yet to fully test that it is 100% working)
- Train and predict methods

Also some basic utilities:
- Functions to calculate accuracy, losses, one-hot encoding, relu, sigmoid and derivatives of these.

**Overall objective**

The ultimate goal is to be able to have a library that covers the types of neural networks that are taught in Christopher Manning's Deep Learning with NLP course:
http://web.stanford.edu/class/cs224n/. This is all for my own learning but hopefully this can help others who are looking to create their own libraries.

**Requirements**

- Python 3.8+
- Poetry

**Setup**

The Poetry library for Python is used for package management. Once the repo has been cloned, in the command line type:

```poetry install```

This should pull all the required libraries (mainly Numpy and Pandas)

A test script, using the MNIST data set (of course) is found in ```mnist_test.py```. The data for this can be easily found online, for example here: http://yann.lecun.com/exdb/mnist/

This data needs to be converted to a standard csv format. Included in the utils is a simple function (courtesy of https://pjreddie.com/projects/mnist-in-csv/).

**To do**

- Cross-entropy loss
- Regularization
- Early stopping
- Dropout
- Convolutional layers
- RNN and LSTM units 