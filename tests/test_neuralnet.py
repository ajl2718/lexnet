from lexnet import __version__
from lexnet.utils import relu, sigmoid 
from lexnet import Net
import numpy as np


def test_version():
    assert __version__ == '0.1.0'


def test_feedforward():
    """
    Test that the feedforward steps work for different
    combinations of weights and architectures
    """
    # define the net architecture
    layers = [4, 2, 1]
    activations = [relu, relu]
    
    weights = [np.array([[1, 2, 0, -1], [-1, -1, 4, 2]]), np.array([[1, -1]])]
    biases = [np.array([[0], [0]]), np.array([[0]])]

    net = Net(layers=layers, activations=activations)
    net.weights = weights
    net.biases = biases 

    input_example = np.array([1, 0, 2, -1]).reshape(-1, 1)

    # feedforward
    Z_layers, A_layers = net.feedforward(input_example)

    assert np.all(Z_layers[0] == np.array([[2], [5]]))
    assert np.all(Z_layers[1] == np.array([[-3]]))
    assert np.all(A_layers[1] == np.array([[2], [5]]))
    assert np.all(A_layers[2] == np.array([[0]]))
