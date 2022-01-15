import numpy as np 
from .utils import sigmoid, dsigmoid, relu 
from .utils import ms_loss, dms_loss
from tqdm import tqdm
from .utils import accuracy

class Net():
    """
    Implements a sequence of fully-connected
    neural net layers
    """
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations

        weights = []
        biases = []

        # to do:  better initialization
        for n in range(1, len(layers)):
            W = np.random.randn(layers[n], layers[n-1]) / np.sqrt(layers[n])
            b = np.random.randn(layers[n], 1)
            weights.append(W)
            biases.append(b)

        self.weights = weights
        self.biases = biases

    def predict(self, X):
        """
        Given a set of input examples, make predictions for the output

        Args:
        X (np.array of shape (num_features, num_examples)): input data

        Return:
        Y_hat (np.array of shape (num_output_features, num_examples)): output predictions
        """
        Y_hat = self.feedforward(X)[-1][-1]
        
        return Y_hat

    def feedforward(self, X):
        """
        Take an input produce the corresponding output 
        along with the values at the intermediate layers

        Args:
        X (np.array of shape (num_features, num_examples)): the input examples

        Return:
        Z_layers (list of np.arrays): the outputs from each layer
        B_layers (list of np.arrays): the output activations from each layer
        """
        Z_new = X
        Z_layers = []
        A_layers = [X]

        for W, B, activation in zip(self.weights, self.biases, self.activations):
            Z_new = np.matmul(W, Z_new) + B
            A_new = activation(Z_new)
            Z_layers.append(Z_new)
            A_layers.append(A_new)

        return Z_layers, A_layers

    def backpropagate(self, X, Y):
        """
        Given an input and output, backpropagate to calculate
        the derivatives of the error with respect to each of the
        weights and biases in the network.

        From: http://neuralnetworksanddeeplearning.com/chap2.html

        Args:
        X (np.array of shape (num_features, num_examples)): input examples
        Y (np.array of shape (num_output_features, num_examples)): input labels

        Return:
        dW_layers (list of np.arrays): the derivatives w.r.t. weights W
        db_layers (list of np.arrays): the derivatives w.r.t. biases b
        """
        # intialize the gradients w.r.t. W and b
        dW_layers = []
        db_layers = []

        # neural network parameters
        weights = self.weights 
        biases = self.biases

        # feedforward step
        z_layers, a_layers = self.feedforward(X)

        # calculate the deltas
        # initial delta (from the output layer)
        # assumes particular loss function and activations. To do: generalise
        delta_final = dms_loss(Y, a_layers[-1]) * dsigmoid(z_layers[-1])
        delta_old = delta_final

        # calculate the initial derivatives w.r.t W and b
        dW = np.matmul(delta_final, a_layers[-2].T)
        db = np.mean(delta_final, axis=1).reshape(-1, 1)

        dW_layers.append(dW)
        db_layers.append(db)

        # calculate the remaining deltas
        for n in range(1, len(weights)):
            W_temp, b_temp = weights[-n], biases[-n]
            delta_new = np.matmul(W_temp.T, delta_old) * dsigmoid(z_layers[-n-1])
            delta_old = delta_new
            dW = np.matmul(delta_new, a_layers[-n-2].T)
            db = np.mean(delta_new, axis=1).reshape(-1, 1)
            dW_layers.append(dW)
            db_layers.append(db)

        # reverse the order so that the first element of each of the 
        # derivatives is the first weight / bias in the network
        return dW_layers[::-1], db_layers[::-1]
        
    def train(self, X, Y, num_epochs, batch_size, epsilon, verbose=False):
        """
        Implements the stochastic gradient descent algorithm
        to find optimal values of the parameters of the neural network
            
        Args:
        X (list of np.array of shape (num_features, num_examples)): the training examples
        y (list of np.array of shape (num_output_features, num_examples)): the training labels
        num_epochs (int): number of epochs to run algorithm for
        batch_size (int): number of examples to use for approximating gradient 
        epsilon (float): the scaling of the gradient
        verbose (bool): print the steps in the training if True
            
        Return:
        losses: the time-series of losses throughout training
        """
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        if isinstance(X, list):
            X_train, X_test = X 
            Y_train, Y_test = Y
        else:
            X_train = X_test = X 
            Y_train = Y_test = Y 

        # split the data up into num_batches based on batch_size
        num_examples = Y_train.shape[1]
        num_batches = int(num_examples / batch_size)
            
        # for each batch in each epoch
        for epoch in range(num_epochs):
            pbar = tqdm(range(num_batches))
            for batch in pbar:
                pbar.set_description(f"Epoch {epoch}")
                X_batch, Y_batch = X_train[:, batch*batch_size:(batch+1)*batch_size], Y_train[:, batch*batch_size:(batch+1)*batch_size]
                dWs, dbs = self.backpropagate(X_batch, Y_batch)
                # update the values of the weights and biases
                for n in range(0, len(self.weights)):
                    self.weights[n] = self.weights[n] - epsilon * dWs[n]
                    self.biases[n] = self.biases[n] - epsilon * dbs[n]
            # calculate the loss
            # model predictions and loss
            Y_preds = self.predict(X_train)
            loss = ms_loss(Y_preds, Y_train)
            y_hat = np.argmax(Y_preds, axis=0).reshape(-1, 1)
            y = np.argmax(Y_train, axis=0).reshape(-1, 1)
            train_accuracy = accuracy(y_hat, y)
            train_accuracies.append(train_accuracy)
            train_losses.append(loss)

            Y_preds = self.predict(X_test)
            loss = ms_loss(Y_preds, Y_test)
            y_hat = np.argmax(Y_preds, axis=0).reshape(-1, 1)
            y = np.argmax(Y_test, axis=0).reshape(-1, 1)
            test_accuracy = accuracy(y_hat, y)
            train_accuracies.append(train_accuracy)
            test_losses.append(loss)
            print(f"Train loss: {train_losses[-1]}\nTest loss: {test_losses[-1]}")
            print(f"Train accuracy: {train_accuracy}\nTest accuracy: {test_accuracy}")

        return train_losses, test_losses, train_accuracies, test_accuracies