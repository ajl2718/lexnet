import numpy as np 

def sigmoid(x):
    """
    Sigmoid activation
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    """
    Derivative of the sigmoid activation
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    Rectified linear unit activation
    """
    result = np.zeros(x.shape)
    index_pos0, index_pos1 = np.where(x >= 0)
    index_neg0, index_neg0 = np.where(x < 0)
    result[index_pos0, index_pos1] = x[index_pos0, index_pos1]
    result[index_neg0, index_neg1] = 0
    return result


def drelu(x):
    """
    Derivative of the relu activation
    """
    result = np.zeros(x.shape)
    index_pos0, index_pos1 = np.where(x >= 0)
    index_neg0, index_neg0 = np.where(x < 0)
    result[index_pos0, index_pos1] = 1
    result[index_neg0, index_neg1] = 0
    return result
    

def cross_entropy_loss(Y, Y_hat):
    """
    Cross-entropy loss of a batch of values

    Args
    Y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    Y_hat (np.array of shape (num_output_features, num_examples)): 
        predicted output value array of shape (output_dim, num_samples)

    Returns
    C (float): mean squared loss for the true and predicted outputs
    """
    C = np.sum(-np.nan_to_num(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)), axis=0).mean()
    return C


def dcross_entropy_loss(Y, Y_hat):
    """
    Derivative of the cross-entropy loss with respect to the 
    prediction variable

    Args
    y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    y_hat (np.array of shape (num_output_features, num_examples)): 
        predicted output value array of shape (output_dim, num_samples)

    Returns
    dC (np.array): the derivative of the mean squared loss function w.r.t. Y_hat
    """
    # to avoid division by zero
    epsilon = 0.001

    dC_num = Y_hat - Y
    dC_denom = Y_hat * (1 - Y_hat) + epsilon
    dC = (dC_num / dC_denom).mean(axis=1).reshape(-1, 1)
    return dC


def ms_loss(Y, Y_hat):
    """
    Mean-squared loss function for a single 
    batch of values

    Args
    Y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    Y_hat (np.array of shape (num_output_features, num_examples)): 
        predicted output value array of shape (output_dim, num_samples)

    Returns
    C (float): mean squared loss for the true and predicted outputs
    """
    D =  Y - Y_hat 
    #C = 0.5 * np.diag(np.matmul(D.T, D)).mean() # too much memory
    C = 0.5 * np.sum(D**2, axis=0).mean()
    return C


def dms_loss(Y, Y_hat):
    """
    Derivative of the mean-squared loss function with respect to 
    the prediction variable

    Args
    y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    y_hat (np.array of shape (num_output_features, num_examples)): 
        predicted output value array of shape (output_dim, num_samples)

    Returns
    dC (np.array): the derivative of the mean squared loss function w.r.t. Y_hat
    """
    dC = (Y_hat - Y)
    return dC


def dloss_function_mapping(loss_function_name):
    """
    Calculate the derivative of the loss function with respect to the
    output activation value

    Args
    loss_function_name (str): the loss function to calculate derivative of
        options are:
        'ms_loss'
        'ce_loss'

    Return
    dloss_fn (func): the derivative of loss_function
    """
    mapping = {'ms_loss': dms_loss, 'ce_loss': dcross_entropy_loss}
    dloss_fn = mapping[loss_function_name]
    return dloss_fn


def loss_function_mapping(loss_function_name):
    """
    Given the string describing the loss function, return
    the actual loss function corresponding to it

    Args
    loss_function_name (str): the loss function to calculate derivative of
        options are:
        'ms_loss'
        'ce_loss'

    Return
    loss_fn (func): the actual loss function
    """
    mapping = {'ms_loss': ms_loss, 'ce_loss': cross_entropy_loss}
    loss_fn = mapping[loss_function_name]
    return loss_fn


def delta(Y, A_final, Z_final, activation, dloss):
    """
    Calculate the 'error in the output neuron' for a given loss function
    and activation. This is used in the recursion of the backprop 
    algorithm to calculate the derivatives of the cost function
    with respect to the weights and biases

    Args:
    Y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    A_final (np.array of shape (num_output_features, num_examples)): 
        predicted output values
    Z_final (np.array of shape (num_output_features, num_examples)):
        predicted output values (pre activation)
    activation (function): 
        the activation function in the final layer of the network
    dloss (function):
        the derivative of the loss function of the network

    Returns:
    delta (np.array of shape (num)output_features, num_examples )): 
        mean squared loss for the true and predicted outputs
    """
    delta = dloss(Y, A_final) * activation(Z_final)
    return delta


def delta_final_ms(Y, A_final, Z_final, activation):
    """
    Calculate the 'error in the output neuron' for mean square loss

    Args:
    Y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    A_final (np.array of shape (num_output_features, num_examples)): 
        predicted output values
    Z_final (np.array of shape (num_output_features, num_examples)):
        predicted output values (pre activation)
    activation (function): 
        the activation function in the final layer of the network

    Returns:
    delta (np.array of shape (num)output_features, num_examples )): 
        mean squared loss for the true and predicted outputs
    """
    delta = dms_loss(Y, A_final) * activation(Z_final)
    return delta


def delta_final_ce(Y, A_final, Z_final, activation):
    """
    Calculate the 'error in the output neuron' for cross-entropy loss

    Args:
    Y (np.array of shape (num_output_features, num_examples)): 
        correct output value one-hot encoded array of shape (output_dim, num_samples)
    A_final (np.array of shape (num_output_features, num_examples)): 
        predicted output values
    Z_final (np.array of shape (num_output_features, num_examples)):
        predicted output values (pre activation)
    activation (function): 
        the activation function in the final layer of the network

    Returns:
    delta (np.array of shape (num)output_features, num_examples )): 
        mean squared loss for the true and predicted outputs
    """

    delta = dcross_entropy_loss(Y, A_final) * activation(Z_final)
    return delta


def grad_descent_simple(num_steps, epsilon, x0, f, df):
    """
    Simple example of gradient descent for cubic function.

    Args:
    num_steps (int): number of iterations to run algorithm for
    epsilon (float): scaling parameter for the jumps.
    x0 (float): initial value of parameter
    f (func): function that we are optimising
    df (func): the derivative of the function that we are optimising

    Return:
    x_best (float), f_best (float), df_best (float): tuple of optimal values
    """

    x1 = x0

    for n in range(0, num_steps):
        x1 = x1 - epsilon * df(x1)
    
    x_best, f_best, df_best = x1, f(x1), df(x1)
    return x_best, f_best, df_best


# convert from the idx format to csv 
# reference: https://pjreddie.com/projects/mnist-in-csv/
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    
    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


def label_to_onehot(label, length):
    """
    Convert a label to a one-hot encoded vector

    # to do: vectorize

    Args:
    label (integer): the input label
    length (integer): the total number of output labels possible

    Return:
    output (np.array of shape (length, 1)): the output one-hot vector
    """
    onehot = np.zeros((length, 1))
    onehot[label, 0] = 1
    return onehot


def accuracy(Y_pred, Y_true):
    """
    Calculate the accuracy of the predictions

    Args:
    Y_pred (np.array of shape (num_examples, )): predicted labels
    Y_true (np.array of shape (num_examples, )): true labels

    Return:
    acc (float): the fraction of labels that have been correctly classified
    """
    Y_pred = Y_pred.reshape(-1, 1)
    Y_true = Y_true.reshape(-1, 1)
    acc = np.mean(Y_pred == Y_true)

    return acc


def precision(Y_pred, Y_true):
    """
    Calculate the precision of the predictions

    Args:
    Y_pred (np.array of shape (num_examples, )): predicted labels
    Y_true (np.array of shape (num_examples, )): true labels

    Return:
    acc (float): the fraction of labels that have been correctly classified
    """
    return 0