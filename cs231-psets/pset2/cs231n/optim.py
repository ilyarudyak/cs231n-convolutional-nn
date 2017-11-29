import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """

    LEARNING_RATE = 'learning_rate'
    MOMENTUM = 'momentum'
    VELOCITY = 'velocity'

    if config is None:
        config = {}
    config.setdefault(LEARNING_RATE, .01)
    config.setdefault(MOMENTUM, .9)
    v = config.get(VELOCITY, np.zeros_like(w))

    next_w = w
    #############################################################################
    # TODO: Implement the momentum update formula. Store the updated value in   #
    # the next_w variable. You should also use and update the velocity v.       #
    #############################################################################
    v = config[MOMENTUM] * v - config[LEARNING_RATE] * dw
    next_w += v
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    config[VELOCITY] = v

    return next_w, config


def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """

    LEARNING_RATE = 'learning_rate'
    DECAY_RATE = 'decay_rate'
    EPS = 'epsilon'
    CACHE = 'cache'

    if config is None: config = {}
    config.setdefault(LEARNING_RATE, 1e-2)
    config.setdefault(DECAY_RATE, 0.99)
    config.setdefault(EPS, 1e-8)
    config.setdefault(CACHE, np.zeros_like(x))

    next_x = None
    #############################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x   #
    # in the next_x variable. Don't forget to update cache value stored in      #
    # config['cache'].                                                          #
    #############################################################################
    config[CACHE] = config[DECAY_RATE] * config[CACHE] + (1 - config[DECAY_RATE]) * (dx ** 2)
    next_x = x - config[LEARNING_RATE] * dx / (np.sqrt(config[CACHE]) + config[EPS])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """

    LEARNING_RATE = 'learning_rate'
    BETA1 = 'beta1'
    BETA2 = 'beta2'
    EPS = 'epsilon'
    M = 'm'
    V = 'v'
    T = 't'

    if config is None: config = {}
    config.setdefault(LEARNING_RATE, 1e-3)
    config.setdefault(BETA1, 0.9)
    config.setdefault(BETA2, 0.999)
    config.setdefault(EPS, 1e-8)
    config.setdefault(M, np.zeros_like(x))
    config.setdefault(V, np.zeros_like(x))
    config.setdefault(T, 0)

    next_x = None
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config.                                                         #
    #############################################################################
    learning_rate, beta1, beta2, eps, m, v, t = config[LEARNING_RATE], config[BETA1], \
        config[BETA2], config[EPS], config[M], config[V], config[T] + 1

    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx ** 2)
    mb = m / (1 - beta1 ** t)
    vb = v / (1 - beta2 ** t)
    next_x = x - learning_rate * mb / (np.sqrt(vb) + eps)

    config[M], config[V], config[T] = m, v, t
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config
