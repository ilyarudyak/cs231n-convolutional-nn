import numpy as np


def toy_data():
    np.random.seed(42)
    W = np.random.random((12, 10))
    X = np.random.randint(0, high=256, size=(5, 12))
    y = np.random.randint(0, high=10, size=(5,))
    return W, X, y


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
        print 'loss_i / N = %.0f' % (loss_i / N)
    loss /= N
    return loss


def svm_loss_no_loops_no_grad(W, X, y):
    N = X.shape[0]
    scores = X.dot(W)

    correct_score = [np.arange(N), y]
    margin = np.maximum(0, scores - scores[correct_score].reshape(N, 1) + 1)
    margin[correct_score] = 0

    loss_i = np.sum(margin, axis=1) / N
    print loss_i
    loss = np.sum(loss_i)
    return loss


def svm_loss_naive_with_grad(W, X, y):
    dW = np.zeros(W.shape)  # grad

    # compute the loss and the gradient
    C = W.shape[1]
    N = X.shape[0]
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # classes_with_positive_margin = 0  # grad
        loss_i = 0.0
        for j in range(C):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss_i += margin
                # classes_with_positive_margin += 1  # grad
                # dW[:, j] += X[i] / num_train  # grad
        loss += loss_i
        print 'loss_i / N = %.2f' % (loss_i / N)
        # dW[:, y[i]] -= classes_with_positive_margin * X[i] / num_train  # grad

    loss /= N

    return loss, dW


if __name__ == '__main__':
    W, X, y = toy_data()
    print 'loss = %.0f' % svm_loss_one_loop_no_grad(W, X, y)
    print 'loss = %.0f' % svm_loss_no_loops_no_grad(W, X, y)
