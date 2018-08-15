import numpy as np


class Net(object):
    def __init__(self, obs_size, action_size):
        k1 = 1 / np.sqrt(obs_size)
        self.weights1 = 2 * k1 * np.random.rand(obs_size, 32) - k1
        self.bias1 = 2 * k1 * np.random.rand(1, 32) - k1

        k2 = 1 / np.sqrt(32)
        self.weights2 = 2 * k2 * np.random.rand(32, action_size) - k2
        self.bias2 = 2 * k2 * np.random.rand(1, action_size) - k2
        
        self.parameters = [self.weights1, self.bias1, self.weights2, self.bias2]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.add(np.matmul(x, self.weights1), self.bias1)
        np.maximum(x, 0, x)
        x = np.add(np.matmul(x, self.weights2), self.bias2)
        return softmax(x, axis=1)


def softmax(X, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: 
        p = p.flatten()

    return p