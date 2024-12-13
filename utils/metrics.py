import numpy as np
from scipy.stats import pearsonr
import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    # u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    # d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    # return (u/d).mean(-1)
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    sse = np.square(pred - true).sum()
    sst = np.square(true - true.mean()).sum()
    r2 = 1 - sse / sst
    # r2_score(true, pred, multioutput='raw_values')
    return r2

def metric(pred, true):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    corr = CORR(pred, true)
    
    return mae, mse, rmse, mape, mspe, r2, corr