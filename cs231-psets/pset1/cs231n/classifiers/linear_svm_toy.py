import numpy as np


def toy_data():
    np.random.seed(42)
    W = np.random.random((12, 10))
    X = np.random.randint(0, high=256, size=(5, 12))
    y = np.random.randint(0, high=10, size=(5,))
    return W, X, y


# -------------- no grad ---------------


def svm_loss_naive_no_grad(W, X, y):
    C = W.shape[1]
    N = X.shape[0]
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_i = 0.0
        for j in range(C):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss_i += margin
        loss += loss_i
        print 'loss_i / N = %.0f' % (loss_i / N)
    loss /= N
    return loss


def svm_loss_one_loop_no_grad(W, X, y):
    N = X.shape[0]
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)
        loss_i = 0.0
        margin = np.maximum(0, scores - scores[y[i]] + 1)
        margin[y[i]] = 0
        loss_i += np.sum(margin)
        loss += loss_i
    loss /= N
    return loss


def svm_loss_no_loops_no_grad(W, X, y):
    N = X.shape[0]
    scores = X.dot(W)

    correct_score = [np.arange(N), y]
    margin = np.maximum(0, scores - scores[correct_score].reshape(N, 1) + 1)
    margin[correct_score] = 0

    loss_i = np.sum(margin, axis=1) / N
    loss = np.sum(loss_i)
    return loss


# -------------- with grad ---------------


def svm_loss_naive_with_grad(W, X, y):
    dW1 = np.zeros(W.shape)  # grad
    dW2 = np.zeros(W.shape)

    # compute the loss and the gradient
    C = W.shape[1]
    N = X.shape[0]
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        classes_with_positive_margin = 0  # grad
        loss_i = 0.0
        for j in range(C):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss_i += margin
                classes_with_positive_margin += 1  # grad
                dW1[:, j] += X[i] / N  # grad
                print
        loss += loss_i
        dW2[:, y[i]] -= classes_with_positive_margin * X[i] / N  # grad

    loss /= N

    return loss, dW1, dW2, dW1 + dW2


def svm_loss_no_loops_with_grad(W, X, y):
    dW = np.zeros(W.shape)
    N = X.shape[0]
    D = X.shape[1]
    scores = X.dot(W)

    correct_score = [np.arange(N), y]
    margins = np.maximum(0, scores - scores[correct_score].reshape(N, 1) + 1)
    margins[correct_score] = 0

    loss_i = np.sum(margins, axis=1) / N
    loss = np.sum(loss_i)

    margins_bin = np.zeros_like(margins)
    margins_bin[margins > 0] = 1
    margins_bin[correct_score] -= np.sum(margins_bin, axis=1)
    dW += X.T.dot(margins_bin) / N

    return loss, dW


if __name__ == '__main__':
    W, X, y = toy_data()
    _, dW1, dW2, dW = svm_loss_naive_with_grad(W, X, y)
    print 'naive grads:'
    print dW1[:2, :3]
    print dW2
    print dW[:2, :3]

    _, dW = svm_loss_no_loops_with_grad(W, X, y)
    print 'vectorized:'
    print dW[:2, :3]
