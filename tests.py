import numpy as np 
from utils import sigmoid, dsigmoid, relu 
from utils import ms_loss, dms_loss, grad_descent_simple 
from net import Net

# set up a basic neural net
layers = [8, 4, 2]
activations = [sigmoid, sigmoid]
net1 = Net(layers, activations)

# input, output example pair
x = np.array([[1, 0, 1, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0]]).reshape(-1, 3)
y = np.array([[0, 1], [1, 0], [1,0]]).reshape(-1, 3)

# feedforward step
z_layers, a_layers = net1.feedforward(x)

# backpropagation step
dW_layers, db_layers = net1.backpropagate(x, y)

# a simple example of gradient descent
f = lambda x: x*(x-2)*(x-4)*(x-6)
df = lambda x: ( (x-2)*(x-4)*(x-6) + x*(x-4)*(x-6) + x*(x-2)*(x-6) + x*(x-2)*(x-4))

# calculate the optimal parameters
x_min, f_min, df_min = grad_descent_simple(1000, 0.001, -5, f, df)

# create some random training data
num_features = 4
num_examples = 1024
X = np.random.randn(num_features, num_examples)
Y = np.random.randn(1, num_examples)

num_epochs = 20
batch_size = 64
epsilon = 1e-3

# gradient descent on the neural net
W_best, b_best, losses = net1.grad_descent(X, y, num_epochs, batch_size, epsilon, verbose=True)