import numpy as np

def prepare_y_hat(y_hat):
    y_hat = np.array(y_hat)
    y_hat = np.vstack([1-y_hat, y_hat]).T
    return y_hat

def repeat_cost_matrix(cost_matrix, n_samples):
    return np.repeat(cost_matrix, n_samples).reshape(2, 2, n_samples).transpose(2, 0, 1)

def create_num_vars(n_samples, n_classes):
    num_vars = []
    for i in range(n_samples):
        num_vars.append([None for _ in range(n_classes)])
    return num_vars
