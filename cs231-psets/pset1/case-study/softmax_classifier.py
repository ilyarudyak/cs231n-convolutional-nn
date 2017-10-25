import numpy as np


class SoftmaxClassifier(object):

    def __init__(self):
        self.W = None
        self.b = None

    def train(self, X, y, step_size=1e-0, reg=1e-3, num_iters=200):

        if self.W is None:
            self.W = 0.01 * np.random.randn(X.shape[1], np.max(y) + 1)
        if self.b is None:
            self.b = np.zeros((1, np.max(y) + 1))

        num_examples = X.shape[0]
        for i in range(num_iters):
            # evaluate class scores, [N x K]
            scores = np.dot(X, self.W) + self.b

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(num_examples), y])
            data_loss = np.sum(corect_logprobs) / num_examples
            reg_loss = 0.5 * reg * np.sum(self.W * self.W)
            loss = data_loss + reg_loss
            if i % 10 == 0:
                print "iteration %d: loss %f" % (i, loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(num_examples), y] -= 1
            dscores /= num_examples

            # backpropate the gradient to the parameters (W,b)
            dW = np.dot(X.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

            dW += reg * self.W  # regularization gradient

            # perform a parameter update
            self.W += -step_size * dW
            self.b += -step_size * db

    def predict(self, X):
        scores = np.dot(X, self.W) + self.b
        return np.argmax(scores, axis=1)



