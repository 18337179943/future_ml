"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from functools import partial
import imp
from tkinter import N
import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
from joblib import Parallel, delayed
from scipy.stats import kurtosis
from scipy.stats import skew
import statsmodels.api as sm
import bottleneck as bn
import talib as tb
import scipy.stats as st


__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    type : int
        The type of function
    """

    def __init__(self, function, name, arity, _type):
        self.function = function
        self.name = name
        self.arity = arity
        self.type = _type

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity, _type='normal'):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    """
    '''
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)
    '''
    return _Function(function, name, arity, _type)

def _apply(x: pd.Series, func, window):
    '''用于对每行做变换'''
    x_i = x.copy()
    x_adj = x_i.dropna()
    x_i[~x_i.isnull()] = x_adj.rolling(window).apply(func).values
    return pd.Series(x_i)

def _df_apply(x, func, window):
    '''apply dataframe'''
    df_x = pd.DataFrame(x)
    df_x = df_x.apply(partial(_apply, func=func, window=window), axis=1)
    return df_x.values

def _apply_x1x2(x1, x2, func):
    x_res = pd.DataFrame(np.repeat(np.nan, x1.shape[0]*x1.shape[1]).reshape(x1.T.shape))
    for i in range(len(x1)):    # 每一行
        x1_i, x2_i = x1[i], x2[i]
        df = pd.DataFrame({'x1': x1_i, 'x2': x2_i})
        # index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        index_i = df_adj.index
        # res = func(df_adj)
        res = _shape_arr(func(df_adj), len(index_i))
        x_res.iloc[index_i, i] = res
    return x_res.values.T

def _apply_xn(x_li, func):
    x_res = pd.DataFrame(np.repeat(np.nan, x_li[0].shape[0]*x_li[0].shape[1]).reshape(x_li[0].T.shape))
    for i in range(len(x_li[0])):    # 每一行
        x_dic = {}
        for j in range(len(x_li)):
            x_dic[f'x{j}'] = x_li[j][i]
        df = pd.DataFrame(x_dic)
        df_adj = df.dropna()
        index_i = df_adj.index
        # res = func(df_adj)
        res = _shape_arr(func(*[df_adj.iloc[:, i].values for i in range(len(x_li))]), len(index_i))
        x_res.iloc[index_i, i] = res
    return x_res.values.T

def _apply_x1(x1, func):
    x_res = pd.DataFrame(np.repeat(np.nan, x1.shape[0]*x1.shape[1]).reshape(x1.T.shape))
    for i in range(len(x1)):    # 每一行
        x1_i = x1[i]
        df = pd.DataFrame({'x1': x1_i})
        index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        # res = func(df_adj)
        res = _shape_arr(func(df_adj), len(index_i))

        x_res.iloc[index_i, i] = res
    return x_res.values.T 

def _apply_xn_d(x_li, func, window, window1=None):
    if window1 is None:
        window1 = window
    x_res = pd.DataFrame(np.repeat(np.nan, x_li[0].shape[0]*x_li[0].shape[1]).reshape(x_li[0].T.shape))
    for i in range(len(x_li[0])):    # 每一行
        x_dic = {}
        for j in range(len(x_li)):
            x_dic[f'x{j}'] = x_li[j][i]
        df = pd.DataFrame(x_dic)
        df_adj = df.dropna()
        index_i = df_adj.index
        res = [func(*[df_adj.iloc[k-window+1:k+1].values[:,l] for l in range(df_adj.shape[1])], window1)[-1] if k >= window -1
                else np.nan for k in range(len(df_adj))]
        res = _shape_arr(np.array(res), len(index_i))
        x_res.iloc[index_i, i] = res
    return x_res.values.T

def _shape_arr(x, shape):
    '''转换矩阵shape，多余部分用nan填充'''
    arr = np.array([np.nan]*shape)
    try:
        arr[-x.shape[0]:] = x
    except:
        print(x.shape, shape)
        print('------------')
    return arr

# 定义函数：参数x1和x2是形状相同的numpy.ndarray（可以是二维），返回两者的协方差
def _cov_np(x1, x2):
    left = x1
    right = x2
    
    # 按前值填充缺失值
    # left = fill_nan(left)
    # right = fill_nan(right)
    
    # 协方差计算
    ldem = left - np.nanmean(left, axis = 0)
    rdem = right - np.nanmean(right, axis = 0)
    num = ldem * rdem
    col_sum = np.nansum(num, axis = 0)
    # col_sum[col_sum == 0] = np.nan
    cov = col_sum / x1.shape[0]
    
    return cov

# 自己定义的函数
def _ts_rank(x, window):
    '''当所在前window个分位数'''
    def quantile(x:pd.Series):
        return x.rank().iloc[-1] / len(x)
    return _df_apply(x, quantile, window)

def _ts_delay(x, window):
    '''window天之前的值'''
    def delay(x):
        return x.iloc[0]
    return _df_apply(x, delay, window)

def _ts_correlation(x1, x2, window):
    '''x1, x2前window个相关系数'''
    x_res = pd.DataFrame(np.repeat(np.nan, x1.shape[0]*x1.shape[1]).reshape(x1.T.shape))
    for i in range(len(x1)):    # 每一行
        x1_i, x2_i = x1[i], x2[i]
        df = pd.DataFrame({'x1': x1_i, 'x2': x2_i})
        index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        res = [df_adj.iloc[j-window:j].corr().iloc[0, 1] if j >= window else np.nan for j in range(len(df_adj))]
        res = _shape_arr(np.array(res), len(index_i))
        
        x_res.iloc[index_i, i] = res
    return x_res.values.T

def _ts_covariance(x1, x2, window):
    '''x1, x2前window个相关系数'''
    x_res = pd.DataFrame(np.repeat(np.nan, x1.shape[0]*x1.shape[1]).reshape(x1.T.shape))
    for i in range(len(x1)):    # 每一行
        x1_i, x2_i = x1[i], x2[i]
        df = pd.DataFrame({'x1': x1_i, 'x2': x2_i})
        index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        res = [_cov_np(df_adj.iloc[j-window:j, 0].values, df_adj.iloc[j-window:j, 1].values) if j >= window else np.nan for j in range(len(df_adj))]
        res = _shape_arr(np.array(res), len(index_i))
        x_res.iloc[index_i, i] = res
    return x_res.values.T

def _ts_scale(x, window):
    '''求当前值相对前window个值的百分比'''
    def scale(x):
        return x.iloc[-1] / len(x)
    return _df_apply(x, scale, window)

def _ts_delta(x, window):
    '''x和前window的x的差值'''
    def delta(x):
        return x.iloc[-1] - x.iloc[0]
    return _df_apply(x, delta, window)

def _ts_decay_linear(x, window):
    '''前window个加权平均'''
    def decay_linear(x):
        return np.mean(x.values*np.arange(1, len(x)+1))
    return _df_apply(x, decay_linear, window)

def _ts_min(x, window):
    '''前win个最小值'''
    return _df_apply(x, np.min, window)

def _ts_max(x, window):
    '''前win个最大值'''
    return _df_apply(x, np.max, window)

def _ts_argmax(x, window):
    '''前win个最大值的索引'''
    return _df_apply(x, np.argmax, window)

def _ts_argmin(x, window):
    '''前win个最小值索引'''
    return _df_apply(x, np.argmin, window)

def _ts_sum(x, window):
    '''前win个值之和'''
    return _df_apply(x, np.sum, window)

def _ts_product(x, window):
    '''前win个值之积'''
    return _df_apply(x, np.prod, window)

def _ts_std(x, window):
    '''前win个值标准差'''
    return _df_apply(x, np.std, window)

def _ts_zscore(x, window):
    '''(X-mean(X, window))/std(X, window)'''
    def zscore(x):
        return (x.iloc[-1]-np.mean(x))/np.std(x)
    return _df_apply(x, zscore, window)

def _ts_skewness(x, window):
    '''前win个峰度'''
    def skew(x):
        return x.skew()
    return _df_apply(x, skew, window)

def _ts_kurtosis(x, window):
    '''前win个峰度'''
    def kurt(x):
        return x.kurt()
    return _df_apply(x, kurt, window)

def _ts_max_diff(x, window):
    '''最大值和x差值'''
    def max_diff(x):
        return x.iloc[-1]-np.max(x)
    return _df_apply(x, max_diff, window)

def _ts_min_diff(x, window):
    '''最小值和x差值'''
    def min_diff(x):
        return x.iloc[-1]-np.min(x)
    return _df_apply(x, min_diff, window)

def _ts_return(x, window):
    '''前win的return'''
    def m_return(x):
        return 0 if x.iloc[0] == 0 else (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
    return _df_apply(x, m_return, window)

def _ts_sharp(x, window):
    '''前win的ir'''
    def sharp(x):
        return 0 if np.std(x) == 0 else np.mean(x) / np.std(x)
    return _df_apply(x, sharp, window)

def _ts_median(x, window):
    '''前win的中位数'''
    return _df_apply(x, np.median, window)

def _ts_zscore_square(x, window):
    def zscore_square(x):
        return 0 if np.std(x) == 0 else ((x.iloc[-1] - np.mean(x)) / np.std(x))**2
    return _df_apply(x, zscore_square, window)

# 2022.9.7 new add
def _ts_mean_return(x, window):
    '''window个x的变化率的'''
    def mean_return(x: pd.Series):
        return x.pct_change().mean()
    return _df_apply(x, mean_return, window)

# ta-lib的函数
def _ts_dema(x, window):
    ''''''
    def dema(x):
        return tb.DEMA(x.values, (len(x)+1)/2)[-1]
    return _df_apply(x, dema, window)

def _ts_kama(x, window):
    ''''''
    def kama(x):
        return tb.KAMA(x.values, len(x)-1)[-1]   
    return _df_apply(x, kama, window)    

def _ts_ma(x, window):
    def ma(x):
        return tb.MA(x.values, len(x))[-1]
    return _df_apply(x, ma, window)

def _ts_midpoint(x, window):
    '''win前最大值和最小值的均值'''
    def midpoint(x):
        return tb.MIDPOINT(x.values, len(x))
    return _df_apply(x, midpoint, window)

def _ts_midprice(x1, x2, window):
    '''mean(max(x1,win), min(x2,win))'''
    return _apply_xn_d([x1, x2], tb.MIDPRICE, window)

def _ts_aroonosc(x1, x2, window):
    return _apply_xn_d([x1, x2], tb.AROONOSC, window, window-1)

def _ts_willr(x1, x2, x3, window):
    return _apply_xn_d([x1, x2, x3], tb.WILLR, window)

def _ts_cci(x1, x2, x3, window):
    return _apply_xn_d([x1, x2, x3], tb.CCI, window)

def _ts_adx(x1, x2, x3, window):
    return _apply_xn_d([x1, x2, x3], tb.ADX, window)

def _ts_mfi(x1, x2, x3, x4, window):
    return _apply_xn_d([x1, x2, x3, x4], tb.MFI, window)

def _ts_natr(x1, x2, x3, window):
    return _apply_xn_d([x1, x2, x3], tb.NATR, window)

def _ts_beta(x1, x2, window):
    def beta(x1, x2, window):
        x1_return = (x1[1:]-x1[:-1]) / x1[:-1]
        x2_return = (x2[1:]-x2[:-1]) / x2[:-1]
        res = st.linregress(x1_return, x2_return)[0]
        return [res]
    return _apply_xn_d([x1, x2], beta, window)

def _ts_linearreg_angle(x, window):
    return _apply_xn_d([x], tb.LINEARREG_ANGLE, window)

def _ts_linearreg_intercept(x, window):
    return _apply_xn_d([x], tb.LINEARREG_INTERCEPT, window)

def _ts_linearreg_slope(x, window):
    return _apply_xn_d([x], tb.LINEARREG_SLOPE, window)

def _ht_dcphase(x):
    def ht_dcphase(df_adj):
        x = df_adj['x1'].values
        return tb.HT_DCPHASE(x)
    return _apply_x1(x, ht_dcphase)

def _neg(x):
    '''相反数'''
    def neg(df_adj):
        x = df_adj['x1'].values
        return -x
    return _apply_x1(x, neg)

def _sigmoid(x):
    '''sigmoid X'''
    def sigmoid(df_adj):
        x = df_adj['x1'].values
        return 1/(1+np.exp(-x))
    return _apply_x1(x, sigmoid)

def _log(x):
    '''log x'''
    def m_log(df_adj):
        x = df_adj['x1'].values
        return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.)
    return _apply_x1(x, m_log)

def _abs(x):
    '''abs x'''
    def m_abs(df_adj):
        return np.abs(df_adj['x1'].values)
    return _apply_x1(x, m_abs)

def _sqrt(x):
    '''sqrt x'''
    def m_sqrt(df_adj):
        x = df_adj['x1'].values
        return np.sqrt(np.abs(x))
    return _apply_x1(x, m_sqrt)

def _sin(x):
    '''sin x'''
    def m_sin(df_adj):
        x = df_adj['x1'].values
        return np.sin(x)
    return _apply_x1(x, m_sin)

def _cos(x):
    '''sin x'''
    def m_cos(df_adj):
        x = df_adj['x1'].values
        return np.cos(x)
    return _apply_x1(x, m_cos)

def _sign(x):
    '''sin x'''
    def m_sign(df_adj):
        x = df_adj['x1'].values
        return np.sign(x)
    return _apply_x1(x, m_sign)

def _add(x1, x2):
    '''add'''
    def m_add(df_adj):
        return np.add(df_adj['x1'].values, df_adj['x2'].values)
    return _apply_x1x2(x1, x2, m_add)

def _reduce(x1, x2):
    ''''''
    def m_reduce(df_adj):
        return df_adj['x1'].values - df_adj['x2'].values
    return _apply_x1x2(x1, x2, m_reduce)

def _multiply(x1, x2):
    ''''''
    def m_multiply(df_adj):
        return df_adj['x1'].values * df_adj['x2'].values
    return _apply_x1x2(x1, x2, m_multiply)

def _division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    def m_division(df_adj):
        x1, x2 = df_adj['x1'].values, df_adj['x2'].values
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)
    return _apply_x1x2(x1, x2, m_division)

# 2022/9/13 更新

def _ts_wma(x, window):
    """
    WMA.
    """
    def wma(x):
        return tb.WMA(x.values, len(x))[-1]
    return _df_apply(x, wma, window)

# def _ts_t3(x, window):
#     """
#     T3.
#     """
#     # def func(x):
#     #     return tb.T3(x.values, timeperiod=len(x), vfactor=0)[-1]
#     return _apply_xn_d([x], tb.T3, window, window-2)

def _ts_trima(x, window):
    """
    TRIMA.
    """
    def func(x):
        return tb.TRIMA(x.values, len(x))[-1]
    return _df_apply(x, func, window)

def _ts_cmo(x, window):
    """
    CMO.
    """
    return _apply_xn_d([x], tb.CMO, window, window-1)

def _ts_mom(x, window):
    """
    MOM.
    """
    return _apply_xn_d([x], tb.MOM, window, window-1)

def _ts_rocr(x, window):
    """
    ROCR.
    """
    return _apply_xn_d([x], tb.ROCR, window, window-1)
    
def _ts_rocp(x, window):
    """
    ROCP.
    """
    return _apply_xn_d([x], tb.ROCP, window, window-1)

def _ts_rocr_100(x, window):
    """
    ROCR100.
    """
    return _apply_xn_d([x], tb.ROCR100, window, window-1)

def _obv(x1, x2):
    """
    OBV.
    """
    def func(df_adj):
        return tb.OBV(df_adj['x1'].values, df_adj['x2'].values)
    return _apply_x1x2(x1, x2, func)

def _ts_rsi(x, window):
    """
    Relative Strenght Index (RSI).
    """
    return _apply_xn_d([x], tb.RSI, window, window-1)

def _ts_dx(x1, x2, x3, window):
    """
    DX.
    """
    return _apply_xn_d([x1, x2, x3], tb.DX, window, window-1)

def _ts_minus_di(x1, x2, x3, window):
    """
    MINUS_DI.
    """
    return _apply_xn_d([x1, x2, x3], tb.MINUS_DI, window, window-1)

def _ts_plus_di(x1, x2, x3, window):
    """
    PLUS_DI.
    """
    return _apply_xn_d([x1, x2, x3], tb.PLUS_DI, window, window-1)

def _trange(x1, x2, x3):
        """
        TRANGE.
        """
        return _apply_xn([x1, x2, x3], tb.TRANGE)

def _avgprice(x1, x2, x3, x4):
    """
    AVGPRICE.
    """
    return _apply_xn([x1, x2, x3, x4], tb.AVGPRICE)

def _medprice(x1, x2):
    """
    MEDPRICE.
    """
    return _apply_xn([x1, x2], tb.MEDPRICE)

def _typprice(x1, x2, x3):
    """
    TYPPRICE.
    """
    return _apply_xn([x1, x2, x3], tb.TYPPRICE)

def _wclprice(x1, x2, x3):
    """
    WCLPRICE.
    """
    return _apply_xn([x1, x2, x3], tb.WCLPRICE)

def _ts_tbbeta(x1, x2, window):
    """贝塔系数"""
    return _apply_xn_d([x1, x2], tb.BETA, window, window-1)

def _ts_correl(x1, x2, window):
    """皮尔逊相关系数"""
    return _apply_xn_d([x1, x2], tb.CORREL, window)

def _ts_linearreg(x, window):
    """线性回归"""
    return _apply_xn_d([x], tb.LINEARREG, window)

def _ts_linerreg_angle(x, window):
    """线性回归的角度"""
    return _apply_xn_d([x], tb.LINEARREG_ANGLE, window)

def _ts_tsf(x, window):
    """时间序列预测"""
    return _apply_xn_d([x], tb.TSF, window)

def _ts_var(x, window):
    """方差"""
    return _apply_xn_d([x], tb.VAR, window)

def _atan(x):
    """反正切值"""
    return _apply_xn([x], tb.ATAN)

def _ceil(x):
    """向上取整数"""
    return _apply_xn([x], tb.CEIL)

def _cosh(x):
    """双曲正弦函数"""
    return _apply_xn([x], tb.COSH)

def _exp(x):
    """指数曲线"""
    return _apply_xn([x], tb.EXP)

def _floor(x):
    """向下取整数"""
    return _apply_xn([x], tb.FLOOR)

def _ln(x):
    """自然对数"""
    return _apply_xn([x], tb.LN)

def _log10(x):
    """对数函数log"""
    return _apply_xn([x], tb.LOG10)

def _sinh(x):
    """双曲正弦函数"""
    return _apply_xn([x], tb.SINH)

def _tan(x):
    """对数函数log"""
    return _apply_xn([x], tb.TAN)

def _ts_minus_dm(x1, x2, window):
    """
    MINUS_DM.
    """
    return _apply_xn_d([x1, x2], tb.MINUS_DM, window)

def _ts_plus_dm(x1, x2, window):
    """
    PLUS_DM.
    """
    return _apply_xn_d([x1, x2], tb.PLUS_DM, window)



def debug_functions():
    i = 4
    d = pd.DataFrame()
    d['w'] = [7854.0,7894,2,5,7,3,2,5,89]*i
    # d['w1'] = [1,67,2,5,7,3,2,5,89]*i
    # d['w2'] = [1,5,243,5,7,3,67,5,89]*i
    # d['w4'] = [77,1,2,5,73,3,2,5,89]*i

    low = d.values.T

    index_name = "_ts_argmin(_cosh(low), 12)"
    index_name = "_cosh(low)"
    val = eval(index_name)
    print(val)

    # debug_functions()



#=======================================================================
ts_delta = make_function(function = _ts_delta,name = 'ts_delta',arity=2,_type='ts')
ts_rank = make_function(function = _ts_rank,name = 'ts_rank',arity=2,_type='ts')
ts_delay = make_function(function = _ts_delay,name = 'ts_delay',arity=2,_type='ts')
ts_scale = make_function(function = _ts_scale,name = 'ts_scale',arity=2,_type='ts')
ts_correlation = make_function(function = _ts_correlation,name = 'ts_correlation',arity=3,_type='ts')
ts_covariance = make_function(function = _ts_covariance,name = 'ts_covariance',arity=3,_type='ts')
ts_decay_linear = make_function(function = _ts_decay_linear,name = 'ts_decay_linear',arity=2,_type='ts')
ts_min = make_function(function=_ts_min, name='ts_min', arity=2, _type='ts')
ts_max = make_function(function=_ts_max, name='ts_max', arity=2, _type='ts')
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=2, _type='ts')
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=2, _type='ts')
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2, _type='ts')
ts_product = make_function(function = _ts_product,name = 'ts_product',arity=2,_type='ts')
ts_std = make_function(function = _ts_std,name = 'ts_std',arity=2,_type='ts')
ts_skewness = make_function(function = _ts_skewness,name = 'ts_skewness',arity=2,_type='ts')
ts_kurtosis = make_function(function = _ts_kurtosis,name = 'ts_kurtosis',arity=2,_type='ts')
sigmoid= make_function(function=_sigmoid,name='sigmoid',arity=1)
ts_max_diff = make_function(function = _ts_max_diff,name = 'ts_max_diff',arity=2,_type='ts')
ts_min_diff = make_function(function = _ts_min_diff,name = 'ts_min_diff',arity=2,_type='ts')
ts_return = make_function(function = _ts_return,name='ts_return',arity=2,_type='ts')
ts_zscore = make_function(function = _ts_zscore,name = 'ts_zscore',arity=2,_type='ts')
ts_scale = make_function(function = _ts_scale,name = 'ts_scale',arity=2,_type='ts')
ts_median = make_function(function = _ts_median,name = 'ts_median',arity=2,_type='ts')
ts_sharp = make_function(function=_ts_sharp, name='ts_sharp', arity=2, _type='ts')
ts_zscore_square = make_function(function=_ts_zscore_square, name='ts_zscore_square', arity=2, _type='ts')
log = make_function(function=_log, name='log', arity=1)
abs = make_function(function=_abs, name='abs', arity=1)
sqrt = make_function(function=_sqrt, name='sqrt', arity=1)
sin = make_function(function=_sin, name='sin', arity=1)
cos = make_function(function=_cos, name='cos', arity=1)
sign = make_function(function=_sign, name='sign', arity=1)
add = make_function(function=_add, name='add', arity=2)
division = make_function(function=_division, name='division', arity=2)

# 2022.9.7 new add
ts_dema = make_function(function = _ts_dema,name = 'ts_dema',arity=2,_type='ts')
ts_mean_return = make_function(function = _ts_mean_return,name = 'ts_mean_return',arity=2,_type='ts')
ts_kama = make_function(function = _ts_kama,name = 'ts_kama',arity=2,_type='ts')
ts_ma = make_function(function = _ts_ma,name = 'ts_ma',arity=2,_type='ts')
ts_midpoint = make_function(function = _ts_midpoint,name = 'ts_midpoint',arity=2,_type='ts')
ts_midprice = make_function(function = _ts_midprice,name = 'ts_midprice',arity=3,_type='ts')
ts_aroonosc = make_function(function = _ts_aroonosc,name = 'ts_aroonosc',arity=3,_type='ts')
ts_willr = make_function(function=_ts_willr, name='ts_willr', arity=4, _type='ts')
ts_cci = make_function(function=_ts_cci, name='ts_cci', arity=4, _type='ts')
ts_adx = make_function(function=_ts_adx, name='ts_adx', arity=4, _type='ts')
ts_mfi = make_function(function=_ts_mfi, name='ts_mfi', arity=5, _type='ts')
ts_natr = make_function(function=_ts_natr, name='ts_natr', arity=4, _type='ts')
ts_beta = make_function(function = _ts_beta,name = 'ts_beta',arity=3,_type='ts')
ts_linearreg_angle = make_function(function = _ts_linearreg_angle,name = 'ts_linearreg_angle',arity=2,_type='ts')
ts_linearreg_intercept = make_function(function = _ts_linearreg_intercept,name = 'ts_linearreg_intercept',arity=2,_type='ts')
ts_linearreg_slope = make_function(function = _ts_linearreg_slope,name = 'ts_linearreg_slope',arity=2,_type='ts')
ht_dcphase = make_function(function = _ht_dcphase,name = 'ht_dcphase',arity=1)
neg = make_function(function = _neg,name = 'neg',arity=1)
multiply = make_function(function=_multiply, name='multiply', arity=2)
reduce = make_function(function=_reduce, name='reduce', arity=2)

# 2022.9.13 new add
ts_wma = make_function(function = _ts_wma,name = 'ts_wma',arity=2,_type='ts')
ts_trima = make_function(function = _ts_trima,name = 'ts_trima',arity=2,_type='ts')
ts_cmo = make_function(function = _ts_cmo,name = 'ts_cmo',arity=2,_type='ts')
ts_mom = make_function(function = _ts_mom,name = 'ts_mom',arity=2,_type='ts')
ts_rocr = make_function(function = _ts_rocr,name = 'ts_rocr',arity=2,_type='ts')
ts_rocp = make_function(function = _ts_rocp,name = 'ts_rocp',arity=2,_type='ts')
ts_rocr_100 = make_function(function = _ts_rocr_100,name = 'ts_rocr_100',arity=2,_type='ts')
obv = make_function(function = _obv,name = 'obv',arity=2)
ts_rsi = make_function(function = _ts_rsi,name = 'ts_rsi',arity=2,_type='ts')
ts_dx = make_function(function = _ts_dx,name = 'ts_dx',arity=4,_type='ts')
ts_minus_di = make_function(function = _ts_minus_di,name = 'ts_minus_di',arity=4,_type='ts')
ts_plus_di = make_function(function = _ts_plus_di,name = 'ts_plus_di',arity=4,_type='ts')
trange = make_function(function = _trange,name = 'trange',arity=3)
avgprice = make_function(function = _avgprice,name = 'avgprice',arity=4)
medprice = make_function(function = _medprice,name = 'medprice',arity=2)
typprice = make_function(function = _typprice,name = 'typprice',arity=3)
wclprice = make_function(function = _wclprice,name = 'wclprice',arity=3)
ts_tbbeta = make_function(function = _ts_tbbeta,name = 'ts_tbbeta',arity=3,_type='ts')
ts_correl = make_function(function = _ts_correl,name = 'ts_correl',arity=3,_type='ts')
ts_linearreg = make_function(function = _ts_linearreg,name = 'ts_linearreg',arity=2,_type='ts')
ts_linerreg_angle = make_function(function = _ts_linerreg_angle,name = 'ts_linerreg_angle',arity=2,_type='ts')
ts_tsf = make_function(function = _ts_tsf,name = 'ts_tsf',arity=2,_type='ts')
ts_var = make_function(function = _ts_var,name = 'ts_var',arity=2,_type='ts')
atan = make_function(function = _atan,name = 'atan',arity=1)
ceil = make_function(function = _ceil,name = 'ceil',arity=1)
cosh = make_function(function = _cosh,name = 'cosh',arity=1)
exp = make_function(function = _exp,name = 'exp',arity=1)
floor = make_function(function = _floor,name = 'floor',arity=1)
ln = make_function(function = _ln,name = 'ln',arity=1)
log10 = make_function(function = _log10,name = 'log10',arity=1)
sinh = make_function(function = _sinh,name = 'sinh',arity=1)
tan = make_function(function = _tan,name = 'tan',arity=1)
ts_minus_dm = make_function(function = _ts_minus_dm,name = 'ts_minus_dm',arity=3,_type='ts')
ts_plus_dm = make_function(function = _ts_plus_dm,name = 'ts_plus_dm',arity=3,_type='ts')



_function_map = {'ts_delta': ts_delta,
                 'ts_rank': ts_rank,
                 'ts_delay': ts_delay,
                 'ts_scale': ts_scale,
                 'ts_correlation': ts_correlation,
                 'ts_covariance': ts_covariance,
                 'ts_decay_linear': ts_decay_linear,
                 'ts_min': ts_min,
                 'ts_max': ts_max,
                 'ts_argmax': ts_argmax,
                 'ts_argmin': ts_argmin,
                 'ts_sum': ts_sum,
                 'ts_product': ts_product,
                 'ts_std': ts_std,
                 'ts_skewness': ts_skewness,
                 'ts_kurtosis': ts_kurtosis,
                 'ts_max_diff': ts_max_diff,
                 'ts_min_diff': ts_min_diff,
                 'ts_return': ts_return,
                 'ts_zscore': ts_zscore,
                 'ts_scale': ts_scale,
                 'ts_median': ts_median,
                 'ts_sharp': ts_sharp,
                 'ts_zscore_square': ts_zscore_square,
                 'log': log,
                 'abs': abs,
                 'sqrt': sqrt,
                #  'sin': sin,
                #  'cos': cos,
                 'sign': sign,
                 'add': add,
                 'division': division,
                 'sigmoid': sigmoid,
                 }


# 2022.9.13 new add
_function_map2 = {'ts_wma': ts_wma,
                  'ts_trima': ts_trima,
                  'ts_cmo': ts_cmo,
                  'ts_rocr': ts_rocr,
                  'ts_mom': ts_mom,
                  'ts_rocp': ts_rocp,
                  'ts_rocr_100': ts_rocr_100,
                  'obv': obv,
                  'ts_rsi': ts_rsi,
                  'ts_dx': ts_dx,
                  'ts_minus_di': ts_minus_di,
                  'ts_plus_di': ts_plus_di,
                  'trange': trange,
                  'avgprice': avgprice,
                  'medprice': medprice,
                  'typprice': typprice,
                  'wclprice': wclprice,
                  'ts_tbbeta': ts_tbbeta,
                  'ts_correl': ts_correl,
                #   'ts_linearreg': ts_linearreg,
                  'ts_linerreg_angle': ts_linerreg_angle,
                #   'ts_tsf': ts_tsf,
                  'ts_var': ts_var,
                  'atan': atan,
                  'ceil': ceil,
                  'cosh': cosh,
                  'exp': exp,
                  'floor': floor,
                  'ln': ln,
                  'log10': log10,
                  'sinh': sinh,
                  'tan': tan,
                  'ts_minus_dm': ts_minus_dm,
                  'ts_plus_dm': ts_plus_dm
}


# 2022.9.7 new add
_function_map1 = {'ts_dema': ts_dema,
                  'ts_mean_return': ts_mean_return,
                  'ts_kama': ts_kama,
                  'ts_ma': ts_ma,
                  'ts_midpoint': ts_midpoint,
                  'ts_midprice': ts_midprice,
                  'ts_aroonosc': ts_aroonosc,
                  'ts_willr': ts_willr,
                  'ts_cci': ts_cci,
                  'ts_adx': ts_adx,
                  'ts_mfi': ts_mfi,
                  'ts_natr': ts_natr,
                  'ts_beta': ts_beta,
                  'ts_linearreg_angle': ts_linearreg_angle,
                  'ts_linearreg_intercept': ts_linearreg_intercept,
                  'ts_linearreg_slope': ts_linearreg_slope,
                  'ht_dcphase': ht_dcphase,
                  'neg': neg,
                  'multiply': multiply,
                  'reduce': reduce
}

_function_map.update(_function_map1)
_function_map.update(_function_map2)

function_set = list(_function_map.keys())
