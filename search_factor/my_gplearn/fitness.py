"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause
import imp
import numbers
import math
import re
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
from scipy.stats import rankdata
import scipy.io as sio
import statsmodels.api as sm
from .utils import winsorize
from scipy import stats
import pandas as pd
__Author__ = 'ZCXY'
from sklearn.feature_selection import mutual_info_regression
import time
import bottleneck as bn
from datas_process.m_datas_process import BaseDataProcess
# import line_profiler


__all__ = ['_Fitness', 'make_fitness', '_rank_ic', '_mutual_info', '_top_return', '_m_spearman', '_total_return', 
           '_total_return_all_quantile', '_cumsum_return_rate', '_cumsum_return_rate1']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        res = self.function(*args)
        if len(pd.DataFrame([res]).dropna()):
            return  res
        else:
            func_name = self.function.__name__
            if func_name == '_rank_ic':
                return 0, 0, 0, 0
            elif func_name == '_total_return_all_quantile':
                return 0, 0, 0
            elif func_name == '_cumsum_return_rate':
                return -1.0
            else:
                return 0

def make_fitness(function, greater_is_better):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    '''
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')
    '''
    return _Fitness(function, greater_is_better)


def mad(df, dm_series = None):
    '''
    中位数法去极值
    '''
    # print('整体去极值')

    def fun(series, dm_series):
        # print(dm_series.loc[:, series.name])
        dm1 = dm_series[series.name].loc['dm1']
        dm = dm_series[series.name].loc['dm']
        # 超过/小于 dm + 5dm1/dm - 5dm1 的修改为 dm + 5dm1/dm - 5dm1
        series[series > dm + 5 * dm1] = dm + 5 * dm1
        series[series < dm - 5 * dm1] = dm - 5 * dm1
        return series
    # try:
    if dm_series == None:
        dm_series = cal_dm_and_d1(df)
    # except:
    #     pass
    if len(df.columns) != len(dm_series.columns):
        print(len(df.columns), len(dm_series.columns), '参数不一致')
    # print(dm_series)
    df = df.apply(fun, args=(dm_series, ))
    return  df

def cal_dm_and_d1(df):
    '''按列计算中位数去极值中的参数
    -----
    df：一般行为时间序列，列为因子
    返回：dataframe， 行为dm，dm1，列为原df列名'''
    def fun(series):
        # se = series.dropna()
        # print('series', np.isnan(series.values))
        se = series
        dm = se.median()
        dm1 = np.abs(se - dm).median()
        return pd.Series([dm, dm1], index = ['dm', 'dm1'])
    # df.to_csv('D:/策略开发/futures_ml/search_factor/my_gplearn/raw_data/y_err.csv')
    return df.apply(fun)

def normalize(y_pred):
    '''adf检验时间序列平稳'''
    df_y_pred = pd.DataFrame(y_pred.T)
    df_y_pred_normalize = df_y_pred.apply(lambda x: (x-x.mean())/x.std())
    return df_y_pred_normalize.values.T


# @profile
def _rank_ic(y, y_pred, w):
    # res_li = []
    bdp = BaseDataProcess()
    count_break = 0
    index_concat = pd.concat([pd.DataFrame(i) for i in y_pred], axis=0, ignore_index=True).dropna().iloc[:, 0]
    concat_li = []
    if bdp.adf_test(index_concat):
        thread_20, thread_80 = index_concat.quantile(.2), index_concat.quantile(.8)
        
        for i in range(len(y)):
            df = pd.DataFrame({'y': y[i], 'y_pred': y_pred[i]}).dropna()
            if len(df) == 0:
                count_break += 1
                if count_break > 10:
                    return [0.0, 0.0, 0, 0]
            else:
                df['y_pred'] = mad(pd.DataFrame(df['y_pred'])).values
                # df = df.rolling(70).apply(lambda x: x.rank().iloc[-1])
                # df = df.apply(lambda x: x.rank())
                df.dropna(inplace=True) 
                if len(df):
                    concat_li.append(df.copy())
                    # corr = df['y'].corr(df['y_pred'])
                    # res_li.append(corr)
                else:
                    count_break += 1
                    if count_break > 10:
                        return [0.0, 0, 0]
        df_concat = pd.concat(concat_li, axis=0, ignore_index=True).dropna().apply(lambda x: x.rank())
        corr = df_concat['y'].corr(df_concat['y_pred'])
        return [abs(corr), corr, thread_20, thread_80]
    else:
        return [0.0, 0.0, 0, 0]

def _total_return(y, y_pred, w):
    '''总收益率，用前70个计算分位数判断信号'''
    bdp = BaseDataProcess()
    res_li = []
    def get_sig(x: pd.Series):
        thread_20, thread_80 = x.quantile(.2), x.quantile(.8)
        x_v = x.iloc[-1]
        if x_v > thread_80:
            sig = 1
        elif x_v < thread_20:
            sig = -1
        else:
            sig = 0
        return sig

    index_concat = pd.concat([pd.DataFrame(i) for i in y_pred], axis=0, ignore_index=True).dropna().iloc[:, 0]
    count_break = 0
    if bdp.adf_test(index_concat):
        for i in range(len(y)):
            df = pd.DataFrame({'y': y[i], 'y_pred': y_pred[i]}).dropna()
            if len(df) == 0:
                count_break += 1
                if count_break > 15: return 0.0
                continue
            df['y_pred'] = df['y_pred'].rolling(70).apply(get_sig)
            df.dropna(inplace=True)
            if len(df) == 0:
                count_break += 1
                if count_break > 15: return 0.0
                continue 
            df['y_pred'].replace(0, None, inplace=True)
            df['y_pred'].fillna(method='ffill', inplace=True)
            sig_count = len(set(df['y_pred'].replace(0, None).dropna().to_list()))
            if sig_count < 2:
                count_break += 1
                if count_break > 15: return 0.0     # 判断合约开仓是否只有开一次
            # df.to_csv('D:/策略开发/futures_ml/search_factor/my_gplearn/raw_data/df.csv', index=False)
            # print('have save df...')
            # input()
            df['res'] = df['y']*df['y_pred']+1
            res_li.append(df['res'].cumprod().iloc[-1])
            # print(res_li[-1])

        return abs(bn.nanmean(res_li))
    else:
        return 0.0

def _total_return_all_quantile(y, y_pred, w):
    '''总收益率 总分位数求信号'''
    bdp = BaseDataProcess()
    res_li = []
    count_break = 0
    index_concat = pd.concat([pd.DataFrame(i) for i in y_pred], axis=0, ignore_index=True).dropna().iloc[:, 0]
    beta_rate_li = []
    if bdp.adf_test(index_concat):
        thread_20, thread_80 = index_concat.quantile(.3), index_concat.quantile(.7) # 趋势 修改 .2 .8
        # thread_20, thread_80 = index_concat.quantile(.8), index_concat.quantile(.2)  # 反转
        # thread_50 = index_concat.quantile(.5)
        # thread_20, thread_80 = thread_50, thread_50
        for i in range(len(y)):
            df = pd.DataFrame({'y': y[i], 'y_pred': y_pred[i]}).dropna()
            if len(df) == 0:
                count_break += 1
                if count_break > 15: return 0.0, 0, 0
                continue
            df['y_sig'] = np.where(df['y_pred']>thread_80, 1, 0)
            df['y_sig'] = np.where(df['y_pred']<thread_20, -1, df['y_sig'])
            # df['y_pred'] = df['y_pred'].rolling(70).apply(get_sig)
            df.dropna(inplace=True)
            if len(df) == 0:
                count_break += 1
                if count_break > 15: return 0.0, 0, 0
                continue 
            df['y_sig'].replace(0, None, inplace=True)
            df['y_sig'].fillna(method='ffill', inplace=True)
            sig_count = len(set(df['y_sig'].replace(0, None).dropna().to_list()))
            if sig_count < 2:
                count_break += 1
                if count_break > 15: return 0.0, 0, 0
            # df.to_csv('D:/策略开发/futures_ml/search_factor/my_gplearn/raw_data/df.csv', index=False)
            # print('have save df...')
            # input()
            df['res'] = df['y']*df['y_sig']+1
            # beta_rate = abs((df['y']+1).cumprod().iloc[-1]-1)       # beta的收益
            cost_rate = 0.00025*np.sum(np.where(df['y_sig']!=df['y_sig'].shift(1), 1, 0))
            # print(cost_rate)
            # beta_rate_li.append(1/beta_rate)
            res_li.append(df['res'].cumprod().iloc[-1]-cost_rate)
            # print(res_li[-1])
        
        # res = np.sum(np.array(res_li)*np.array(beta_rate_li))/np.sum(beta_rate_li)

        return [bn.nanmean(res_li), thread_20, thread_80]
    else:
        return 0.0, 0, 0

def _cumsum_return_rate(y, y_pred, w):
    '''根据因子值排序，再对收益率进行累加，
    把最值点和末端的均值的差值与基准的相差百分比作为适应度'''
    df_info = pd.read_csv(f'{pa_prefix}/search_factor/my_gplearn/raw_data/df_fitness_info.csv')
    bdp = BaseDataProcess()
    index_concat = pd.concat([pd.DataFrame(i) for i in y_pred], axis=0, ignore_index=True).dropna().iloc[:, 0]
    df_li = []
    fitness_li = []
    if bdp.adf_test(index_concat):
        for i, j in zip(y, y_pred):
            df_li.append(pd.DataFrame({'y': i, 'y_pred': j}).dropna())

        for i in range(len(df_info)):
            if i == 0:
                start_n, end_n = 0, df_info['contract_count'].iloc[i]
            else:
                start_n, end_n = df_info['contract_count'].iloc[i-1], df_info['contract_count'].iloc[i]

            df_concat = pd.concat(df_li[start_n: end_n])
            df_concat = df_concat.sort_values('y_pred', ascending=True)
            df_concat['y_cumsum'] = df_concat['y'].cumsum()
            if len(df_concat):
                fitness_v = df_concat['y_cumsum'].abs().max() - abs(df_concat['y_cumsum'].iloc[-1])/2
                # fitness_v = df_concat['y_cumsum'].abs().max()
                fitness_li.append((fitness_v - df_info['mean_fitness'].iloc[i]) / df_info['mean_fitness'].iloc[i])
        return bn.nanmean(fitness_li) if len(fitness_li) == len(df_info) else -1.0
    else:
        return -1.0

def _cumsum_return_rate1(y, y_pred, w):
    '''根据因子值排序，再对收益率进行累加，
    把最值点/最后收益率求和作为适应度'''
    df_info = pd.read_csv(f'{pa_prefix}/search_factor/my_gplearn/raw_data/df_fitness_info.csv')
    bdp = BaseDataProcess()
    index_concat = pd.concat([pd.DataFrame(i) for i in y_pred], axis=0, ignore_index=True).dropna().iloc[:, 0]
    df_li = []
    fitness_li = []
    if bdp.adf_test(index_concat):
        for i, j in zip(y, y_pred):
            df_li.append(pd.DataFrame({'y': i, 'y_pred': j}).dropna())

        for i in range(len(df_info)):
            if i == 0:
                start_n, end_n = 0, df_info['contract_count'].iloc[i]
            else:
                start_n, end_n = df_info['contract_count'].iloc[i-1], df_info['contract_count'].iloc[i]

            df_concat = pd.concat(df_li[start_n: end_n])
            df_concat = df_concat.sort_values('y_pred', ascending=True)
            df_concat['y_cumsum'] = df_concat['y'].cumsum()
            if len(df_concat):
                fitness_v = df_concat['y_cumsum'].abs().max()
                fitness_li.append(abs(fitness_v / df_concat['y_cumsum'].iloc[-1]))
        return bn.nanmean(fitness_li) if len(fitness_li) else -1.0
    else:
        return -1.0

def _m_spearman(y, y_pred, w):
    ''''''
    res_li = []
    for i in range(len(y)):
        df = pd.DataFrame({'y': y[i], 'y_pred': y_pred[i]}).dropna()
        if len(df) == 0:
            pass
            # print('alert: _m_spearman got df length 0', i)
            # res_li.append(0)
        else:
            # print('df', np.isnan(df.values))
            # df.reset_index(drop=True, inplace=True)
            df['y_pred'] = mad(pd.DataFrame(df['y_pred'])).values
            res_li.append(stats.spearmanr(df.values)[0])

            # except:
            #     print(len(df), i)
            #     print(type(df))
            #     input()
    return abs(bn.nanmean(res_li))


def _mutual_info(y, y_pred, p, w, neutral_fac):
    import math
#    from sklearn.feature_selection import mutual_info_regression
    
    # 1. 数据检查 & 去极值
    df_y_pred = pd.DataFrame()
    df_y_pred['y_pred'] = y_pred
    df_y_pred['y_pred'] = winsorize(df_y_pred['y_pred'])
    df_y_check = df_y_pred.dropna()
    if df_y_check.shape[0] == 0:
        return 0
    
    # 2. 中性化
    factor_style = pd.DataFrame()
    factor_style['log_mkt'] = neutral_fac[0][0][0][:, p-1]
    factor_style['turn'] = neutral_fac[0][0][2][:, p-1]
    factor_style['std'] = neutral_fac[0][0][3][:, p-1]
    factor_style['return'] = neutral_fac[0][0][4][:, p-1]
    factor_style['fac0'] = neutral_fac[0][0][5][:, p-1]
    factor_style['fac1'] = neutral_fac[0][0][6][:, p-1]
    factor_style['fac2'] = neutral_fac[0][0][7][:, p-1]
    factor_indus_dummy = pd.get_dummies(neutral_fac[0][0][1][:, p-1])
    factor_neutral = pd.concat([factor_style, factor_indus_dummy], axis=1)

    try:
        model = sm.OLS(df_y_pred,factor_neutral, missing='drop')
        results = model.fit()
    except:
        return 0

    #factor_output = results.resid
    # 取残差为中性化后的因子
    factor_output = pd.Series(index=df_y_pred.index)
    factor_output[results.fittedvalues.index] = df_y_pred.iloc[results.fittedvalues.index, 0] - results.fittedvalues
    
    # 3. 计算MI
    df = pd.DataFrame()
    df['y'] = y
    df['y_pred'] = factor_output
    df = df.dropna()
    MI = mutual_info_regression(df[['y_pred']], df['y'], discrete_features=False, n_neighbors=5)[0] # n_neighbors可能需要调优

    if math.isnan(MI):
        MI = 0
    
    #print('p=',p,'MI=',MI)
    #input()
    return MI	


def _top_return(y, y_pred, p, w, neutral_fac, old_top_weight=None, old_bot_weight=None, old_base_weight=None):
    fee = 0.0015
    df_y_pred = pd.DataFrame()
    df_y_pred['y_pred'] = y_pred
    df_y_check = df_y_pred.dropna()
    if df_y_check.shape[0] == 0:
        return 0

    factor_style = pd.DataFrame()
    factor_style['log_mkt'] = neutral_fac[0][0][0][:, p - 1]
    factor_style['turn'] = neutral_fac[0][0][2][:, p - 1]
    factor_style['std'] = neutral_fac[0][0][3][:, p - 1]
    factor_style['return'] = neutral_fac[0][0][4][:, p - 1]
    factor_style['fac0'] = neutral_fac[0][0][5][:, p - 1]
    factor_style['fac1'] = neutral_fac[0][0][6][:, p - 1]
    # factor_style['fac2'] = neutral_fac[0][0][7][:, p - 1]
    factor_indus_dummy = pd.get_dummies(neutral_fac[0][0][1][:, p - 1])
    factor_neutral = pd.concat([factor_style, factor_indus_dummy], axis=1)

    try:
        model = sm.OLS(df_y_pred,factor_neutral, missing='drop')
        results = model.fit()
    except:
        return 0

    # factor_output = results.resid
    # 取残差为中性化后的因子
    factor_output = pd.Series(index=df_y_pred.index)
    factor_output[results.fittedvalues.index] = df_y_pred.iloc[results.fittedvalues.index, 0] - results.fittedvalues

    # 计算准备
    df = pd.DataFrame()
    df['y'] = y
    df['y_pred'] = factor_output
    df['old_top_weight'] = old_top_weight
    df['old_bot_weight'] = old_bot_weight
    df['old_base_weight'] = old_base_weight

    # 1. top
    df['y_pred_pct'] = df['y_pred'].rank() / df['y_pred'].rank().max()
    df['top_weight'] = 0.0
    df.loc[df['y_pred_pct'] > 0.9, 'top_weight'] = 1
    df['top_weight'] = df['top_weight'] / df['top_weight'].sum()
    df['untraded'] = np.isnan(neutral_fac[0][0][0][:,p])
    top_unsold_mask = (df['old_top_weight'] > 0) & (df['y_pred_pct'] <= 0.9) & df['untraded']   # 找出卖不出的股票
    df.loc[top_unsold_mask, 'top_weight'] = df.loc[top_unsold_mask, 'old_top_weight']  # 找出卖不出的股票权重保持不变
    df.loc[df['y_pred_pct'] > 0.9, 'top_weight'] = df.loc[df['y_pred_pct'] > 0.9, 'top_weight']*(1 - df.loc[top_unsold_mask, 'top_weight'].sum())   # 调整本期入选的股票权重
    top_turn = (df['top_weight'] - df['old_top_weight']).abs().sum()
    raw_top_return = (df['top_weight'] * df['y']).sum() # 未扣除交易费用
    Top_return = (1 - top_turn * fee) * (1 + raw_top_return) - 1
    final_top_weight = df['top_weight'] * (df['y'] + 1)
    final_top_weight = final_top_weight / final_top_weight.sum()

    # 2. bot
    df['y_pred_pct'] = df['y_pred'].rank() / df['y_pred'].rank().max()
    df['bot_weight'] = 0.0
    df.loc[df['y_pred_pct'] <= 0.1, 'bot_weight'] = 1
    df['bot_weight'] = df['bot_weight'] / df['bot_weight'].sum()
    df['untraded'] = np.isnan(neutral_fac[0][0][0][:, p])
    bot_unsold_mask = (df['old_bot_weight'] > 0) & (df['y_pred_pct'] > 0.1) & df['untraded']
    df.loc[bot_unsold_mask, 'bot_weight'] = df.loc[bot_unsold_mask, 'old_bot_weight']
    df.loc[df['y_pred_pct'] <= 0.1, 'bot_weight'] = df.loc[df['y_pred_pct'] <= 0.1, 'bot_weight'] * (1 - df.loc[bot_unsold_mask, 'bot_weight'].sum())  # 调整本期入选的股票权重
    bot_turn = (df['bot_weight'] - df['old_bot_weight']).abs().sum()
    raw_bot_return = (df['bot_weight'] * df['y']).sum()  # 未扣除交易费用
    Bot_return = (1 - bot_turn * fee) * (1 + raw_bot_return) - 1
    final_bot_weight = df['bot_weight'] * (df['y'] + 1)
    final_bot_weight = final_bot_weight / final_bot_weight.sum()

    # 3. Base
    df['base_weight'] = 0.0
    df.loc[df['y_pred_pct'] >= 0.0, 'base_weight'] = 1
    df['base_weight'] = df['base_weight'] / df['base_weight'].sum()
    base_unsold_mask = (df['old_base_weight'] > 0) & (np.isnan(df['y_pred_pct'])) & df['untraded']
    base_unsold_mask = base_unsold_mask.values
    df.loc[base_unsold_mask, 'base_weight'] = df.loc[base_unsold_mask, 'old_base_weight']  # 找出卖不出的股票权重保持不变
    df.loc[df['y_pred_pct'] >= 0.0, 'base_weight'] = df.loc[df['y_pred_pct'] >= 0.0, 'base_weight'] * (1 - df.loc[base_unsold_mask, 'base_weight'].sum())  # 调整本期入选的股票权重
    base_turn = (df['base_weight'] - df['old_base_weight']).abs().sum()
    raw_base_return = (df['base_weight'] * df['y']).sum()  # 未扣除交易费用
    base_return = (1 - base_turn * fee) * (1 + raw_base_return) - 1
    final_base_weight = df['base_weight'] * (df['y'] + 1)
    final_base_weight = final_base_weight / final_base_weight.sum()

    return Top_return, Bot_return, base_return, final_top_weight, final_bot_weight, final_base_weight

def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.

def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)


# weighted_pearson = make_fitness(function=_weighted_pearson,
#                                 greater_is_better=True)
# weighted_spearman = make_fitness(function=_weighted_spearman,
#                                  greater_is_better=True)
# mean_absolute_error = make_fitness(function=_mean_absolute_error,
#                                    greater_is_better=False)
# mean_square_error = make_fitness(function=_mean_square_error,
#                                  greater_is_better=False)
# root_mean_square_error = make_fitness(function=_root_mean_square_error,
#                                       greater_is_better=False)
# log_loss = make_fitness(function=_log_loss, greater_is_better=False)
#
# rank_ic = make_fitness(function=_rank_ic, greater_is_better=True)
#
# mutual_info = make_fitness(function=_mutual_info, greater_is_better=True)
#
# top_return = make_fitness(function=_top_return, greater_is_better=True)
#
# _fitness_map = {'pearson': weighted_pearson,
#                 'spearman': weighted_spearman,
#                 'mean absolute error': mean_absolute_error,
#                 'mse': mean_square_error,
#                 'rmse': root_mean_square_error,
#                 'log loss': log_loss,
#                 'rank_ic' : rank_ic,
# 				'mutual_info' : mutual_info,
#                 'top_return': top_return}
