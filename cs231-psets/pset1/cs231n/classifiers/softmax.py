import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, C = X.shape[0], W.shape[1]

    for i in range(N):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # numerical stability (see pp. 16-17 of LC notes)
        correct_score = scores[y[i]]
        denom = np.sum(np.exp(scores))
        for j in range(C):
            pj = np.exp(scores[j]) / denom
            if j == y[i]:
                dW[:, j] += (pj - 1) * X[i]
            else:
                dW[:, j] += pj * X[i]
        loss -= np.log(np.exp(correct_score) / denom)

    dW /= N
    loss /= N

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W ** 2)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, C = X.shape[0], W.shape[1]
    scores = X.dot(W)  # (N, C)
    scores -= np.max(scores)  # numerical stability (see pp. 16-17 of LC notes)
    correct_score = scores[range(N), y]  # (N,)

    denom = np.sum(np.exp(scores), axis=1)  # (N,)
    loss -= np.sum(np.log(np.exp(correct_score) / denom))  # term inside sum: (N,)
    loss /= N

    pj = np.exp(scores) / denom.reshape(N, 1)  # (N, C)
    pj[range(N), y] -= 1
    dW += X.T.dot(pj)  # (D, N).dot(N, C) = (D, C)
    dW /= N

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
