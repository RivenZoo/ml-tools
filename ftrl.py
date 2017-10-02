# coding: utf-8

import math

alpha = 0.01
beta = 1
l1 = 0.1
l2 = 1


def exp(v):
    v = 20 if v > 20 else v
    v = -20 if v < -20 else v
    return math.exp(v)


def sigmod(v):
    return 1 / (1 + exp(-v))


class FTRL:

    def __init__(self, alpha, beta, l1, l2, dimension):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.dimension = dimension
        self._z = [.0] * dimension
        self._q = [.0] * dimension

    def weight(self):
        return [self.get_weight(i) for i in range(self.dimension)]

    def get_weight(self, idx):
        zi = self._z[idx]
        sign = 1
        if zi < 0:
            sign = -1
        if sign * zi < self.l1:
            return 0
        else:
            return (self.l1 * sign - zi) / \
                   (self.l2 + (self.beta + math.sqrt(self._q[idx])) /
                    self.alpha)

    @staticmethod
    def grad(score, xi, y):
        return (score - y) * xi

    def fit(self, X, Y):
        for m in range(len(X)):
            x = X[m]
            y = Y[m]
            score = self.score(x)
            grad = score - y
            G = [grad * x[i] for i in range(self.dimension)]
            for i in range(self.dimension):
                gi = G[i]
                # only handle x[i] != 0
                if gi == 0:
                    continue
                w = self.get_weight(i)
                g2 = math.pow(gi, 2)
                sigma = (math.sqrt(self._q[i] + g2) -
                         math.sqrt(self._q[i])) / self.alpha
                self._z[i] += (gi - sigma * w)
                self._q[i] += g2

    def score(self, x):
        score = sum([self.get_weight(i) * x[i] for i in range(self.dimension)])
        return sigmod(score)

    def predict(self, X):
        ret = []
        for m in range(len(X)):
            x = X[m]
            ret.append(self.score(x))
        return ret


if __name__ == '__main__':
    ftrl = FTRL(alpha, beta, l1, l2, dimension=4)
    from sklearn import datasets
    from sklearn.metrics import roc_auc_score

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # we only take the first two features.
    Y = iris.target
    X = X[:100, :]
    Y = Y[:100]
    # print X
    ftrl.fit(X, Y)
    print ftrl.weight()
    pred = ftrl.predict(X)
    print roc_auc_score(Y, pred)
    print Y[:10], pred[:10]
    x1 = X
    x1[:, 0] *= 0.1
    # print x1
    model = FTRL(alpha, beta, l1, l2, dimension=4)
    model.fit(x1, Y)
    print model.weight()
    pred = ftrl.predict(X)
    print roc_auc_score(Y, pred)
    print Y[:10], pred[:10]
