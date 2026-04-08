import numpy as np


# Trimmed Mean Robust Aggregation
# Removes highest and lowest fraction before averaging
def trimmed_mean(layer_weights, trim_ratio=0.1):
    weights = np.stack(layer_weights)
    n_clients = weights.shape[0]
    trim = int(trim_ratio * n_clients)

    if trim == 0:
        return np.mean(weights, axis=0)

    sorted_weights = np.sort(weights, axis=0)
    trimmed = sorted_weights[trim : n_clients - trim]
    return np.mean(trimmed, axis=0)
