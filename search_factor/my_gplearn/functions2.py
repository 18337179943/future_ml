"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
from joblib import Parallel, delayed
from scipy.stats import kurtosis
from scipy.stats import skew
import statsmodels.api as sm
import bottleneck as bn


def ts_argmax(x1, window):
    interval = x1.shape[1]
    def _df_ts_argmax(k):   
        return np.nanargmax(x1[:, k:window + k], axis=1)+1.0
    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_argmax)(k+1) for k in range (0, interval-window))).T
    result = np.full([x1.shape[0], window],np.nan)
    result = np.column_stack([result,tmparray])     
    return result 
    
    
def ts_argmin(x1, window):
    interval = x1.shape[1]
    def _df_ts_argmax(k):   
        return np.nanargmin(x1[:, k:window + k], axis=1)+1.0
    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_argmax)(k+1) for k in range (0, interval-window))).T
    result = np.full([x1.shape[0], window],np.nan)
    result = np.column_stack([result,tmparray])     
    return result

def delay(x1, window):
    result = x1[:,:-window+1]
    fill = np.full([x1.shape[0], window-1],np.nan)
    result = np.column_stack([fill,result])
    return result

def delta(x1, window):
    result = x1[:,window-1:] / x1[:,:-window+1] - 1
    fill = np.full([x1.shape[0], window-1],np.nan)
    result = np.column_stack([fill,result])
    return result

def ts_stddev(x1, window):
    return bn.move_std(x1, window, ddof = 1)

def ts_sum(x1, window):
    return bn.move_sum(x1, window)

def ts_max(x1, window):
    return bn.move_max(x1, window)

def ts_min(x1, window):
    return bn.move_min(x1, window)

def ts_nanmean(x1, window):    
    return bn.move_mean(x1, window)
    
def ts_prod(x1, window):  
    interval = x1.shape[1]
    def _df_ts_prod(k):   
        return np.nanprod(x1[:, k:window + k], axis=1)
    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_prod)(k+1) for k in range (0, interval-window))).T
    result = np.full([x1.shape[0], window],np.nan)
    result = np.column_stack([result,tmparray])
    result[np.isnan(x1)] = np.nan
    return result

def ts_rank(x1, window):
    interval = x1.shape[1]
    def _df_ts_rank(k):        
        data = pd.DataFrame(x1[:, k:window + k])
        return data.rank(axis = 1).iloc[:,-1].values/window
    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_rank)(k+1) for k in range (0, interval-window))).T
    result = np.full([x1.shape[0], window],np.nan)
    result = np.column_stack([result,tmparray])     
    return result

def rank(x1):
    data = pd.DataFrame(x1)
    rank = data.rank().values
    return rank/np.nanmax(rank, axis = 0)

# 函数功能为按前值填充nan并返回
def fill_nan(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out

# 定义函数：参数x1和x2是形状相同的numpy.ndarray（可以是二维），返回两者的协方差
def cov_np(x1, x2):
    left = x1
    right = x2
    
    # 按前值填充缺失值
    left = fill_nan(left)
    right = fill_nan(right)
    
    # 协方差计算
    ldem = left - np.nanmean(left, axis = 0)
    rdem = right - np.nanmean(right, axis = 0)
    num = ldem * rdem
    col_sum = np.nansum(num, axis = 0)
    col_sum[col_sum == 0] = np.nan
    cov = col_sum / x1.shape[0]
    
    return cov


def ts_covariance(x1, x2, window):      
    result = np.full([x1.shape[0], x1.shape[1]], np.nan)
    for k in range (1, x1.shape[1]-window+1):
        data1 = x1[:, k:window + k].T
        data2 = x2[:, k:window + k].T       
        result[:,k+window-1] = cov_np(data1, data2)   # 调用函数计算data1与data2的协方差
    return result

def corr_np(x1, x2):
    left = x1
    right = x2
    
    # 按前值填充缺失值
    left = fill_nan(left)
    right = fill_nan(right)
    
    # 相关系数计算
    ldem = left - np.nanmean(left, axis = 0)
    rdem = right - np.nanmean(right, axis = 0)
    num = ldem * rdem
    col_sum = np.nansum(num, axis = 0)
    col_sum[col_sum == 0] = np.nan
    dom = (x1.shape[0] - 1) * np.nanstd(left, axis=0, ddof=1) * np.nanstd(right, axis=0, ddof=1)
    corr = col_sum / dom
    
    return corr

def ts_correlation(x1,x2,window):    
    result = np.full([x1.shape[0], x1.shape[1]], np.nan)  
    for k in range (1, x1.shape[1]-window+1):
        data1 = x1[:, k:window + k].T
        data2 = x2[:, k:window + k].T           
        result[:,k+window-1] = corr_np(data1, data2)
    
    return result

def protected_rank_add(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank,axis = 0)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank,axis = 0)

    return  np.add(rank1, rank2)

def protected_rank_sub(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank, axis = 0)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank, axis = 0)

    return  np.subtract(rank1, rank2)

def protected_rank_mul(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank, axis = 0)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank, axis = 0)

    return  np.multiply(rank1, rank2)

def protected_rank_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    data = pd.DataFrame(x1)
    rank = data.rank().values
    rank1 = rank / np.nanmax(rank, axis = 0)
    data = pd.DataFrame(x2)
    rank = data.rank().values
    rank2 = rank / np.nanmax(rank, axis = 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(rank2) > 0.001, np.divide(rank1, rank2), 1.)


def protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def protected_add(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    return  np.add(x1, x2)

def protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def decay_linear(x1, window):
    interval = x1.shape[1]
    num = np.array(list(range(window))) + 1.0
    coe = np.tile(num, (x1.shape[0],1))
    def _sub_decay_linear(k, coe):
        data = x1[:, k:window + k]
        isnan = np.isnan(data)
        coe[isnan] = np.nan
        sum_days = np.nansum(coe,axis = 1)
        sum_days = np.tile(sum_days,(window,1)).T
        coe = coe/sum_days
        decay = np.nansum(coe*data,axis = 1)
        decay[isnan[:,-1]] = np.nan
        return decay
    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_sub_decay_linear)(k + 1, coe) for k in range (0, interval-window))).T
    result = np.full([x1.shape[0], window],np.nan)
    result = np.column_stack([result,tmparray])
    return result

def signedpower(x1, power):
    return np.sign(x1)*np.power(np.abs(x1),power)

#=================================================================================

def ts_skewness(x1, window):
    interval = x1.shape[1]

    def _df_ts_skewness(k):
        return skew(x1[:, k:window + k], axis=1, bias=False)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_skewness)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_kurtosis(x1, window):
    interval = x1.shape[1]

    def _df_ts_kurtosis(k):
        return kurtosis(x1[:, k:window + k], axis=1, bias=False)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_kurtosis)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_max_diff(x1, window):
    interval = x1.shape[1]

    def _df_ts_max_diff(k):
        return x1[:, window + k - 1] - np.nanmax(x1[:, k:window + k], axis=1)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_max_diff)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_min_diff(x1, window):
    interval = x1.shape[1]

    def _df_ts_min_diff(k):
        return x1[:, window + k - 1] - np.nanmin(x1[:, k:window + k], axis=1)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_min_diff)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_return(x1, window):
    interval = x1.shape[1]

    def _df_ts_return(k):
        return (x1[:, window + k - 1] - x1[:, k]) / x1[:, k]

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_return)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_zscore(x1, window):
    interval = x1.shape[1]

    def _df_ts_zscore(k):
        return ((x1[:, window + k - 1]).T - np.nanmean(x1[:, k:window + k], axis=1)) / np.nanstd(x1[:, k:window + k],
                                                                                                 axis=1)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_zscore)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_scale(x1, window):
    interval = x1.shape[1]

    def _df_ts_scale(k):
        return (x1[:, window + k - 1] - np.nanmin(x1[:, k:window + k], axis=1)) / (
                    np.nanmax(x1[:, k:window + k], axis=1) - np.nanmin(x1[:, k:window + k], axis=1))

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_scale)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_min_max_cps(x1, window, f=2):
    interval = x1.shape[1]

    def _df_ts_cps(k):
        return np.nanmin(x1[:, k:window + k], axis=1) + np.nanmax(x1[:, k:window + k], axis=1) - f * x1[:, window + k - 1]

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_cps)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_ir(x1, window):
    interval = x1.shape[1]

    def _df_ts_ir(k):
        return np.nanmean(x1[:, k:window + k], axis=1) / np.nanstd(x1[:, k:window + k], axis=1)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_ir)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result


def ts_median(x1, window):
    interval = x1.shape[1]

    def _df_ts_median(k):
        return np.nanmedian(x1[:, k:window + k], axis=1)

    tmparray = np.array(Parallel(n_jobs=-1)(delayed(_df_ts_median)(k + 1) for k in range(0, interval - window))).T
    result = np.full([x1.shape[0], window], np.nan)
    result = np.column_stack([result, tmparray])
    return result

def sign(x1):
    return np.sign(x1)

#=======================================================================
# 2019-07-11 新增

def zscore(x1):
    zs = (x1 - np.nanmean(x1, axis=0)) / np.nanstd(x1, axis=0)
    return  zs

def zscore_square(x1):
    zs = (x1 - np.nanmean(x1, axis=0)) / np.nanstd(x1, axis=0)
    return  zs**2

def winsorize(x1):

    interval = x1.shape[1]
    data = pd.DataFrame(x1)

    def _is_dummy(x):
        # return x in [0,1,np.nan]
        return (x==0) or (x==1) or (x==np.nan)

    def _winsorize_by_col(k):
        col = x1[:, k].copy()
        m = np.nanmedian(col)
        mm = np.nanmedian(np.abs(col - m))
        if mm == 0:
            m_mask = (col == m)
            col[m_mask] = np.nan
            mm = np.abs(col - m)
            try:
                col[col > m + 5 * mm] = m + 5 * mm
                col[col < m - 5 * mm] = m - 5 * mm
            except ValueError:
                pass
            col[m_mask] = m
        else:
            try:
                col[col > m + 5 * mm] = m + 5 * mm
                col[col < m - 5 * mm] = m - 5 * mm
            except ValueError:
                pass
        return col

    result = np.array(Parallel(n_jobs=-1)(delayed(_winsorize_by_col)(k) for k in range(0, interval))).T

    return result


def regress_resid(y1, x1):

    interval = x1.shape[1]

    def _regress_resid_cs(k):
        y_c = y1[:, k]
        x_c = x1[:, k]
        resid = np.full(x1.shape[0], np.nan)
        try:
            model = sm.OLS(y_c, x_c, missing='drop')
            results = model.fit()
            nanmask = np.isnan(x_c)
            nanmask[np.isnan(y_c)] = True
            resid[~nanmask] = y_c[~nanmask] - results.fittedvalues
        except:
            resid[:] = np.nan
        return resid

    result = np.array(Parallel(n_jobs=-1)(delayed(_regress_resid_cs)(k) for k in range(0, interval))).T

    return result

def non_linear(x1):
    x1 = _zscore(_winsorize(x1))+1
    nl = _regress_resid(x1**3, x1)
    # nl = _winsorize(x1)
    return nl

