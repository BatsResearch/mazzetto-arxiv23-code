import dynamic_algorithm
import numpy as np


def estimate_accuracy(C, k):
    n = len(C[0])
    best_abs = -1
    i_k = 0
    j_k = 0
    for i in range(n):
        for j in range(n):
            if (i != k) and (j != k) and (i != j):
                if np.abs(C[i][j]) > best_abs:
                    i_k = i
                    j_k = j
                    best_abs = np.abs(C[i, j])
    if C[i_k][j_k] == 0:
        acc = 0.5
    else:
        acc = (1 + np.sqrt(np.abs(C[i_k][k] * C[j_k][k] / C[i_k][j_k]))) / 2
    if acc > 0.9:
        acc = 0.9
    acc = np.log(acc / (1 - acc))
    return acc


def estimate_accuracies(votes):
    assert len(set(np.unique(votes).tolist()).difference({-1, 1})) == 0
    C = dynamic_algorithm.build_covariance(votes)
    probs = [estimate_accuracy(C, i) for i in range(votes.shape[1])]
    probs = np.asarray(probs)
    return probs


def predict(votes, weights):
    weighted_votes = np.multiply(votes.astype(np.float32), weights[None, :]).sum(axis=1)
    preds = np.sign(weighted_votes)
    preds[weighted_votes == 0] = np.random.choice(
        [-1, 1], [np.sum(weighted_votes == 0)]
    )
    return preds
