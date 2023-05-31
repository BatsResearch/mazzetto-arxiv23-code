import numpy as np


def compute_best_history_range(
    votes, delta, beta, coef=1, r_base=2, r_power=20, min_ws=None, max_ws=None
):
    R = [int(r_base**i) for i in range(r_power)]

    k = len(R)
    n = len(votes[0])
    best_r = None
    corr_dist_list = list()
    thresh_list = list()
    hist_len_list = list()

    max_time_steps, num_lfs = votes.shape
    idx = 0
    corr_matrix = build_covariance(votes[max_time_steps - R[idx] : max_time_steps])
    while idx + 1 < k and R[idx + 1] < len(votes):
        corr_matrix_next = build_covariance(
            votes[(max_time_steps - R[idx + 1]) : max_time_steps]
        )
        th_val = threshold(R[idx], R[idx + 1], k, n, beta, delta)
        corr_matrix_dists = np.abs(corr_matrix - corr_matrix_next).max()
        # print(R[idx], corr_matrix_dists, th_val )
        # print(corr_matrix)
        # print(corr_matrix_next)
        hist_len_list.append(R[idx])
        corr_dist_list.append(corr_matrix_dists)
        thresh_list.append(th_val * coef)

        if corr_matrix_dists <= (th_val * coef):
            idx = idx + 1
        else:
            if best_r is None:
                best_r = R[idx]
            idx = idx + 1
            # break

        # corr_matrix = 0.5 * corr_matrix + 0.5 * corr_matrix_next
        corr_matrix = corr_matrix_next
    if best_r is None:
        best_r = R[idx - 1]
    if min_ws:
        best_r = max(min_ws, best_r)
    if max_ws:
        best_r = min(max_ws, best_r)
    return {
        "correlation_matrix": corr_matrix,
        "optimal_window_size": best_r,
        "correlation_distances": corr_dist_list,
        "thresholds": thresh_list,
        "window_sizes": hist_len_list,
    }


def threshold(r_1, r_2, k, n, beta, delta):
    C = np.sqrt(2 * np.log((2 * k - 1) * n * (n - 1) / delta))
    return C * (2 * beta / np.sqrt(r_1) + np.sqrt((1 - r_1 / r_2) / r_1))


def build_covariance(X):
    if isinstance(X, list):
        X = np.asarray(X)
    cm = (
        np.einsum("bij,bjh->bih", X[:, :, None], X[:, None, :]).sum(axis=0) / X.shape[0]
    )
    return cm
