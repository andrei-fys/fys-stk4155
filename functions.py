import numpy as np


'''
SOme useful functions for R2, Bias2, Variance
'''

def mse(y_excact, y_predict, axis=0):
    assert y_excact.shape == y_predict.shape
    return np.mean((y_excact - y_predict)**2, axis=axis)


def R2(y_excact, y_predict, axis=None):
    mse_excact_pred = np.sum((y_excact - y_predict)**2, axis=axis)
    variance_excact = np.sum((y_excact - np.mean(y_excact))**2)
    return 1.0 - mse_excact_pred/variance_excact


def bias2(y_excact, y_predict, axis=0):
    return np.mean((y_predict - np.mean(y_excact, keepdims=True, axis=axis))**2)


def ridge_regression_variance(X, sigma2, lmb):
    XT_X = X.T @ X
    W_lmb = XT_X + lmb * np.eye(XT_X.shape[0])
    W_lmb_inv = np.linalg.inv(W_lmb)
    return np.diag(sigma2 * W_lmb_inv @ XT_X @ W_lmb_inv.T)



