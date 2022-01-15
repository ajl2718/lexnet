# test out neural network on MNIST
import numpy as np 
import pandas as pd 

from lexnet.utils import sigmoid, dsigmoid, relu 
from lexnet.utils import ms_loss, dms_loss, grad_descent_simple 
from lexnet.utils import label_to_onehot, accuracy

from lexnet import Net

# MNIST data folder
source_folder = '/home/alex/Desktop/Data/MNIST'

# load up the CSV files for the MNIST
df_train = pd.read_csv(f'{source_folder}/mnist_train.csv', header=None)
df_test = pd.read_csv(f'{source_folder}/mnist_test.csv', header=None)

# extract the values for training and test sets and convert to np.array
X_train, y_train = df_train.values[:, 1:].T, df_train.values[:, 0]
X_test, y_test = df_test.values[:, 1:].T, df_test.values[:, 0]

# scale the image values
X_train = X_train / 255
X_test = X_test / 255

# convert the labels to one-hot vectors
# training data
Y_train = np.concatenate([label_to_onehot(label, 10) for label in y_train], axis=1)
# test data
Y_test = np.concatenate([label_to_onehot(label, 10) for label in y_test], axis=1)

# neural network hyper parameters
layers = [784, 64, 10]
activations = [sigmoid, sigmoid, sigmoid]
net1 = Net(layers, activations)
num_epochs = 16
batch_size = 64
epsilon = 5e-3

# gradient descent on the neural net
losses = net1.train(X_train, Y_train, num_epochs, batch_size, epsilon)

# now make some predictions
y_test_preds = net1.predict(X_test)
y_hat = np.argmax(y_test_preds, axis=0).reshape(-1, 1)
y = y_test.reshape(-1, 1)

# test set accuracy
acc_test = accuracy(y_hat, y)