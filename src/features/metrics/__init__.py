import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def mda(y_true, y_pred):
    return np.mean((np.sign(y_true[1:] - y_pred[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]).astype(int)))

def calculate_metrics(y_real, y_pred):
    metrics = {
        "MAPE": mean_absolute_percentage_error(y_real, y_pred),
        "sMAPE": smape(y_real, y_pred),
        "MAE": mean_absolute_error(y_real, y_pred),
        "MDA": mda(y_real, y_pred)
    }
    return metrics