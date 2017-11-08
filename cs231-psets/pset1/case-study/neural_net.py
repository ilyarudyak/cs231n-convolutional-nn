import numpy as np


class NeuralNetClassifier(object):

    def __init__(self):
        self.h = 100  # size of hidden layer
        self.W, self.W2 = None, None
        self.b, self.b2 = None, None
        self.L = None

    def train(self, X, y, step_size=1e-0, reg=1e-3, num_iters=10000):
        D, K = 2, 3
        num_examples = X.shape[0]

        self.W = 0.01 * np.random.randn(D, self.h)
        self.b = np.zeros((1, self.h))
        self.W2 = 0.01 * np.random.randn(self.h, K)
        self.b2 = np.zeros((1, K))
        self.L = np.zeros((num_iters / 1000, 2))

        for i in xrange(num_iters):
            # evaluate class scores, [N x K]
            hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b)  # note, ReLU activation
            scores = np.dot(hidden_layer, self.W2) + self.b2

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(num_examples), y])
            data_loss = np.sum(corect_logprobs) / num_examples
            reg_loss = 0.5 * reg * np.sum(self.W * self.W) + 0.5 * reg * np.sum(self.W2 * self.W2)
            loss = data_loss + reg_loss
            if i % 1000 == 0:
                self.L[i / 1000, 0], self.L[i / 1000, 1] = i, loss
                print "iteration %d: loss %f" % (i, loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(num_examples), y] -= 1
            dscores /= num_examples

            # backpropate the gradient to the parameters
            # first backprop into parameters W2 and b2
            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            # next backprop into hidden layer
            dhidden = np.dot(dscores, self.W2.T)

            # backprop the ReLU non-linearity
            dhidden[hidden_layer <= 0] = 0
            # finally into W,b
            dW = np.dot(X.T, dhidden)
            db = np.sum(dhidden, axis=0, keepdims=True)

            # add regularization gradient contribution
            dW2 += reg * self.W2
            dW += reg * self.W

            # perform a parameter update
            self.W += -step_size * dW
            self.b += -step_size * db
            self.W2 += -step_size * dW2
            self.b2 += -step_size * db2

    def predict(self, X):
        hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        return np.argmax(scores, axis=1)

