from pickle import FROZENSET
from tkinter import SEL
import matplotlib
matplotlib.use("TkAgg")
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date, datetime
from m_base import datetime_to_str, timestamp_to_datetime
from datas_process.m_futures_factors import MainconInfo, SymbolsInfo
import talib as tb

__Author__ = 'ZCXY'

class Zigzag:
    def __init__(self) -> None:
        self.PEAK = 1
        self.VALLEY = -1


    def identify_initial_pivot(self, X, up_thresh, down_thresh):
        x_0 = X[0]
        x_t = x_0

        max_x = x_0
        min_x = x_0

        max_t = 0
        min_t = 0

        up_thresh += 1
        down_thresh += 1

        for t in range(1, len(X)):
            x_t = X[t]

            if x_t / min_x >= up_thresh:
                return self.VALLEY if min_t == 0 else self.PEAK  # √型, (min_t != 0), initial点算波峰

            if x_t / max_x <= down_thresh:
                return self.PEAK if max_t == 0 else self.VALLEY

            if x_t > max_x:
                max_x = x_t
                max_t = t

            if x_t < min_x:
                min_x = x_t
                min_t = t

        t_n = len(X)-1
        return self.VALLEY if x_0 < X[t_n] else self.PEAK # 始终没有超过up,down的thresh时，根据 initial点 和 final点 算波峰波谷


    def peak_valley_pivots(self, X, up_thresh, down_thresh):
        """
        Find the peaks and valleys of a series.

        :param X: the series to analyze
        :param up_thresh: minimum relative change necessary to define a peak
        :param down_thesh: minimum relative change necessary to define a valley
        :return: an array with 0 indicating no pivot and -1 and 1 indicating
            valley and peak


        The First and Last Elements
        ---------------------------
        The first and last elements are guaranteed to be annotated as peak or
        valley even if the segments formed do not have the necessary relative
        changes. This is a tradeoff between technical correctness and the
        propensity to make mistakes in data analysis. The possible mistake is
        ignoring data outside the fully realized segments, which may bias
        analysis.
        """
        if down_thresh > 0:
            raise ValueError('The down_thresh must be negative.')

        initial_pivot = self.identify_initial_pivot(X, up_thresh, down_thresh)

        t_n = len(X)
        pivots = np.zeros(t_n, dtype=np.int_)
        trend = -initial_pivot
        last_pivot_t = 0
        last_pivot_x = X[0]

        pivots[0] = initial_pivot

        # Adding one to the relative change thresholds saves operations. Instead
        # of computing relative change at each point as x_j / x_i - 1, it is
        # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
        # This saves (t_n - 1) subtractions.
        up_thresh += 1
        down_thresh += 1

        for t in range(1, t_n):
            x = X[t]
            r = x / last_pivot_x

            # 已经计过波峰，进入下跌趋势后
            if trend == -1:
                # 一旦从之前的最低点last_pivot_x，上涨超过up_thresh，
                # 则计之前的最低点为波谷
                if r >= up_thresh:
                    pivots[last_pivot_t] = trend
                    trend = self.PEAK
                    last_pivot_x = x
                    last_pivot_t = t
                # 每次下跌，都计一个最低点last_pivot_x
                elif x < last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t
            else:
                if r <= down_thresh:
                    pivots[last_pivot_t] = trend
                    trend = self.VALLEY
                    last_pivot_x = x
                    last_pivot_t = t
                elif x > last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t

        if last_pivot_t == t_n-1:
            pivots[last_pivot_t] = trend
        elif pivots[t_n-1] == 0:
            pivots[t_n-1] = -trend

        return pivots


    def max_drawdown(self, X):
        """
        Compute the maximum drawdown of some sequence.

        :return: 0 if the sequence is strictly increasing.
            otherwise the abs value of the maximum drawdown
            of sequence X
        """
        mdd = 0
        self.peak = X[0]

        for x in X:
            if x > self.peak:
                self.peak = x

            dd = (self.peak - x) / self.peak

            if dd > mdd:
                mdd = dd

        return mdd if mdd != 0.0 else 0.0


    def pivots_to_modes(self, pivots):
        """
        Translate pivots into trend modes.

        :param pivots: the result of calling ``peak_valley_pivots``
        :return: numpy array of trend modes. That is, between (VALLEY, PEAK] it
        is 1 and between (PEAK, VALLEY] it is -1.
        """

        modes = np.zeros(len(pivots), dtype=np.int_)
        mode = -pivots[0]

        modes[0] = pivots[0]

        for t in range(1, len(pivots)):
            x = pivots[t]
            if x != 0:
                modes[t] = mode
                mode = -x
            else:
                modes[t] = mode

        return modes


    def compute_segment_returns(self, X, pivots):
        """
        :return: numpy array of the pivot-to-pivot returns for each segment."""
        pivot_points = X[pivots != 0]
        return pivot_points[1:] / pivot_points[:-1] - 1.0


    def plot_pivots(self, date, X, pivots):
        # plt.xlim(0, len(X))
        plt.xlim(date[0], date[len(date)-1])
        plt.ylim(X.min()*0.99, X.max()*1.01)
        plt.plot(date, X, 'k:', alpha=0.5)
        plt.plot(date[pivots != 0], X[pivots != 0], 'k-')
        plt.scatter(date[pivots == 1], X[pivots == 1], color='g')
        plt.scatter(date[pivots == -1], X[pivots == -1], color='r')


    def plot_trend(self, date, X, trading_day, pivots, trend):
        plt.xlim(date[0], date[len(date)-1])
        plt.ylim(X.min()*0.99, X.max()*1.01)
        plt.plot(date, X, 'k:', alpha=0.5)
        plt.scatter(date[pivots == 1], X[pivots == 1], color='g')
        plt.scatter(date[pivots == -1], X[pivots == -1], color='r')

        for i in trend.index:
            start = trend.loc[i, 'trading_day_start']
            end = trend.loc[i, 'trading_day_end']
            plt.plot(date[(start<=trading_day) & (trading_day<=end)], 
                    X[(start<=trading_day) & (trading_day<=end)], 'k-')
        plt.show()


class ZigZagInfo():
    '''品种的趋势和震荡'''
    def __init__(self) -> None:
        self.zigzag = Zigzag()
        self.zigzag_params = self.get_zigzag_params()

    def get_trend_shock_segment(self, symbol, startdate, enddate):
        '''获取趋势段和震荡段
        return: 
            trading_day_start trading_day_end    return
        1         2019-01-04      2019-03-04  0.385918
        5         2019-03-26      2019-04-08  0.138776
        6         2019-04-08      2019-05-09 -0.155445
        9         2019-05-27      2019-07-01  0.135640
        '''
        mainconinfo = MainconInfo()
        zigzag_param1, zigzag_param2 = self.zigzag_params[symbol].iloc[0], self.zigzag_params[symbol].iloc[1]
        data = mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=0, is_concat=1)
        data['trading_day'] = data['datetime'].apply(lambda x: datetime_to_str(x).split(' ')[0])
        trading_day = np.array(data['trading_day'])
        date_time = np.array(data['datetime'])
        X = np.array(data['close'])
        # 通过zigzag获取波峰波谷
        pivots = self.zigzag.peak_valley_pivots(X, zigzag_param1, -zigzag_param1)
        df = pd.DataFrame(data={
            'trading_day_end': trading_day[pivots!=0][1:],
            'return': self.zigzag.compute_segment_returns(X, pivots)
            })
        df['trading_day_start'] = df['trading_day_end'].shift(1)
        # 获取趋势段
        df_trend = df[['trading_day_start', 'trading_day_end', 'return']][np.abs(df['return'])>zigzag_param2]
        df_shock = df[['trading_day_start', 'trading_day_end', 'return']][np.abs(df['return'])<=zigzag_param2]
        return df_trend, df_shock

    def seperate_trend_shock_data(self, symbol, df_0: pd.DataFrame):
        '''将数据按趋势项和震荡项区分开, 可用于机器学习的样本数据分类'''
        df = df_0.copy()
        startdate, enddate = timestamp_to_datetime(df['datetime'].iloc[0]), timestamp_to_datetime(df['datetime'].iloc[-1])
        startdate, enddate = datetime(startdate.year, startdate.month, startdate.day), datetime(enddate.year, enddate.month, enddate.day)
        df_trend, df_shock = self.get_trend_shock_segment(symbol, startdate, enddate)
        df['trading_day'] = df['datetime'].apply(lambda x: datetime_to_str(x).split(' ')[0])
        trend_res_li = []
        for i in range(len(df_trend)):
            start, end = df_trend[['trading_day_start', 'trading_day_end']].iloc[i]
            trend_res_li.append(df[(df['trading_day']>=start) & (df['trading_day']<end)])
        df_trend_res = pd.concat(trend_res_li).dropna()
        df_shock_res = df[~df.index.isin(df_trend_res.index.copy())].dropna()
        df_trend_res.reset_index(drop=True, inplace=True), df_shock_res.reset_index(drop=True, inplace=True)
        return df_trend_res, df_shock_res

    def get_zigzag_params(self, startdate=None, enddate=None, save=0):
        '''每个品种对应一组zigzag参数，通过atr/close求参数比例'''
        # pa_prefix = 'D:/策略开发/futures_ml/datas'
        save_pa = f'{pa_prefix}/datas/zigzag_info/'
        if startdate is None:
            startdate, enddate = datetime(2016, 1, 1), datetime(2019, 5, 1)
            try:
                df = pd.read_csv(f'{pa_prefix}/datas/zigzag_info/zigzag_params.csv')
                return df
            except:
                pass
        symbol_li = SymbolsInfo().symbol_li
        mainconinfo = MainconInfo()
        symbol_dic = {}
        for symbol in symbol_li:
            sy_k_line_li = mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=20, load_pa=None, is_concat=0, contract_name=0)
            sy_atr_rate = np.mean([tb.ATR(df_i['high'].values, df_i['low'].values, df_i['close'].values, len(df_i)-1)[-1] / df_i['close'].iloc[-1] 
                            for df_i in sy_k_line_li])
            symbol_dic[symbol] = sy_atr_rate
        
        basic_rate = symbol_dic['RB']

        for symbol in symbol_li:
            symbol_dic[symbol] = [0.04/basic_rate*symbol_dic[symbol], 0.1/basic_rate*symbol_dic[symbol]]

        df = pd.DataFrame(symbol_dic)
        if save:
            makedir(save_pa)
            df.to_csv(f'{save_pa}zigzag_params.csv', index=False)
        return df


def get_trend_segment(symbol='RB', data=None, plot=0, is_sep=0):
    '''
    获取历史中各趋势段
    -----------
    return: 
        trading_day_start trading_day_end    return
    1         2019-01-04      2019-03-04  0.385918
    5         2019-03-26      2019-04-08  0.138776
    6         2019-04-08      2019-05-09 -0.155445
    9         2019-05-27      2019-07-01  0.135640
    '''
    # symbol = 'RB'
    # plot = 1
    # start, end = datetime(2016, 1, 1), datetime(2019, 5, 1)
    # start, end = datetime(2020, 5, 1), datetime(2020, 10, 29)
    zzinfo = ZigZagInfo()
    if is_sep:
        zigzag_param1, zigzag_param2 = zzinfo.zigzag_params[symbol].iloc[0], zzinfo.zigzag_params[symbol].iloc[1]
    else:
        zigzag_param1, zigzag_param2 = 0.04, 0.1
    # print(symbol, zigzag_param1, zigzag_param2)
    start, end = timestamp_to_datetime(data['datetime'].iloc[0]), timestamp_to_datetime(data['datetime'].iloc[-1])
    start, end = datetime(start.year, start.month, start.day), datetime(end.year, end.month, end.day)
    mainconinfo = MainconInfo()
    # symbolinfo = SymbolsInfo()
    # symbol_li = symbolinfo.symbol_li
    zigzag = Zigzag()
    
    trend_dic = {}

    # for symbol in ['SN']:
    df_li = mainconinfo.get_main_contact_k_line(symbol, start, end, delay=0, is_concat=1)
    # trend_dic.update({symbol: []})
    # print(symbol)
    data = df_li
    # print(data)
    # print(type(data))
    # data = data[['datetime', 'close']]
    # print(data)
    # print(type(data['datetime'].iloc[0]), type(data['close'].iloc[0]))
    # input()
    data['trading_day'] = data['datetime'].apply(lambda x: datetime_to_str(x).split(' ')[0])
    # data['trading_day'] = data['datetime'].apply(lambda x: x.split(' ')[0])
    trading_day = np.array(data['trading_day'])
    date_time = np.array(data['datetime'])
    X = np.array(data['close'])

    # 通过zigzag获取波峰波谷
    pivots = zigzag.peak_valley_pivots(X, zigzag_param1, -zigzag_param1)
    df = pd.DataFrame(data={
        'trading_day_end': trading_day[pivots!=0][1:],
        'return': zigzag.compute_segment_returns(X, pivots)
        })
    df['trading_day_start'] = df['trading_day_end'].shift(1)
    # df.dropna(inplace=True)
    # 获取趋势段
    trend = df[['trading_day_start', 'trading_day_end', 'return']][np.abs(df['return'])>zigzag_param2]
    if plot:
        zigzag.plot_trend(date_time, X, trading_day, pivots, trend)
    # trend_dic[symbol].append(trend)
    # return trend_dic
    return trend


def run_get_zigzag_params():
    zzinfo = ZigZagInfo()
    df = zzinfo.get_zigzag_params(save=1)
    print('run_get_zigzag_params done.')

if __name__ == "__main__":

    run_get_zigzag_params()

    # load_pa = f'{pa_prefix}/datas/backtest_res/RB/y_pred_[5, 0.5, 1, 1]_RB_60m_1.2_sample_10_1_return_rate_60m_train_analyze'
    # symbol = 'RB'
    # data = pd.read_csv(f'{load_pa}.csv')
    # data['datetime'] = pd.to_datetime(data['datetime'])
    # # get_trend_segment(symbol, data, 1)
    # zzif = ZigZagInfo()
    # df_trend_res, df_shock_res = zzif.seperate_trend_shock_data(symbol, data)

    # load_pa = f'{pa_prefix}/datas/backtest_res/RB/y_pred_[5, 0.5, 1, 1]_RB_60m_1.2_sample_10_1_return_rate_60m_train_analyze'
    
    # # load_pa = f'{pa_prefix}/datas/ml_result/symbol_result_adj/params/[5, 0.5, 1, 1]_JD_60m_1.3_sample_20_1_return_rate_60m/y_pred_[5, 0.5, 1, 1]_JD_60m_1.3_sample_20_1_return_rate_60m_test_analyze'
    # data = pd.read_csv(f'{load_pa}.csv')
    # data['datetime'] = pd.to_datetime(data['datetime'])
    # print((data['datetime'].iloc[-1] - data['datetime'].iloc[0]).days)
    # # get_trend_segment('RB', data, 1)