import math
import random

def train(X_train, y_train, X_val, y_val, n_classes):
    n_features = len(X_train[0])

    W = [[0.0] * n_features for _ in range(n_classes)]
    counts = [0] * n_classes

    for x, y in zip(X_train, y_train):
        counts[y] += 1
        for j in range(n_features):
            W[y][j] += x[j]

    for c in range(n_classes):
        for j in range(n_features):
            W[c][j] /= counts[c]

    b = [
        -0.5 * sum(v * v for v in W[c])
        for c in range(n_classes)
    ]

    def predict(X):
        preds = []

        for x in X:
            scores = [
                sum(x[j] * W[c][j] for j in range(n_features)) + b[c]
                for c in range(n_classes)
            ]

            preds.append(max(range(n_classes), key=lambda c: scores[c]))

        return preds

    return predict