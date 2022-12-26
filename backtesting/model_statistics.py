from operator import is_
import statistics
from graphql import InputValueDefinitionNode
from matplotlib.colors import NoNorm
import pandas as pd

import numpy as np
import sys

# sys_name = 'windows'
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
# sys.path.insert(0, pa_sys)
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/' if sys_name == 'windows' else '/home/ZhongCheng/futures_ml_linux/'
pa_prefix = '.' if sys_name == 'windows' else '/home/ZhongCheng/futures_ml_linux'
sys.path.insert(0, pa_sys)
from functools import partial
from m_base import *
from datas_process.m_futures_factors import SymbolsInfo
from m_base import m_plot_hist, str_to_datetime
from datas_process.zigzag import *
import shutil


class ModelStatistics:
    '''
    统计模型结果：
    1、信号的胜率
    2、盈亏比
    3、交易次数
    4、最长持仓周期
    5、平均持仓周期
    6、涨跌胜率
    '''
    def __init__(self) -> None:
        '''
        df.columns = [datetime, close, signal, pos, profit, cost, pnl, pnl_cost]
        '''
        self.model_li = ['train', 'val', 'test']
        self.symbolinfo = SymbolsInfo()
        self.df_symbols_all = self.symbolinfo.df_symbols_all
        self.is_sep = 0
        # self.symbol_size = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]['size'].iloc[0]

    def get_train_val_test_pa(self, train_pa):
        '''获取文件路径'''
        val_pa = train_pa.replace('train', 'val')
        test_pa = train_pa.replace('train', 'test')
        return train_pa, val_pa, test_pa

    def _caculate_each_pnl(self, df, symbol_size, return_df=0):
        '''计算每次成交的盈亏'''  # 把res_rate换成了res
        res, holding_period, res_rate, dt_li = [], [], [], []
        pnl_cost, close_price, pos, dt = df['pnl_cost'], df['close'], df['pos'], df['datetime'] 
        df_s = df[df['signal']!=0]
        
        if abs(df_s['signal'].sum()) == len(df_s):
            res.append(pnl_cost.iloc[-1] - pnl_cost.iloc[0])
            holding_period.append(len(df))
            dt_li.append(dt.iloc[0])
        else:
            ind_signal = df_s.index.values

            # df['is_change_pos'] = df['pos']*df['pos'].shift(1)
            # ind_signal = df[df['is_change_pos']<0].index.values 
            for i in range(len(ind_signal)-1):
                # print(len(ind_signal), '---------')
                ind_signal_i = ind_signal[i]
                res.append(pnl_cost.iloc[ind_signal[i+1]] - pnl_cost.iloc[ind_signal_i])
                # hand = pos.iloc[ind_signal_i] if pos.iloc[ind_signal_i] != 0 else pos.iloc[ind_signal[i+1]]
                # res_rate.append(res[-1]*1_000_000/(symbol_size*close_price.iloc[ind_signal_i]*hand))  # 
                res_rate.append(res[-1]*1_000_000/symbol_size/close_price.mean())
                holding_period.append(ind_signal[i+1]-ind_signal_i)
                dt_li.append(dt.iloc[ind_signal[i+1]])

        if return_df:
            df_res = pd.DataFrame({'res': res, 'holding_period': holding_period, 'res_rate': res, 'datetime': dt_li})
            return df_res
        return np.array(res), np.array(holding_period), np.array(res), np.array(dt_li)

    def _caculate_signal_pl(self, df: pd.DataFrame, symbol_size):
        '''四种信号，计算每个信号的盈亏情况'''
        signal_class = ['trend', 'revers', 'adjust', 'other']
        res, holding_period, res_rate, signal = [], [], [], []
        ind_signal = df[df['signal']!=0].index.values
        pnl_cost = df['pnl_cost']
        close_price = df['close']
        signal_class = df['signal_class']
        for i in range(len(ind_signal)-1):
            res.append(pnl_cost.iloc[ind_signal[i+1]] - pnl_cost.iloc[ind_signal[i]])
            res_rate.append(res[-1]*1_000_00/symbol_size/close_price.mean())  # 
            signal.append(signal_class.iloc[ind_signal[i]])
            holding_period.append(ind_signal[i+1]-ind_signal[i])
        df_signal = pd.DataFrame({'res': res, 'holding_period': holding_period, 'res_rate': res_rate, 'signal_class': signal})
        return np.array(res), np.array(holding_period), np.array(res_rate), df_signal

    def caculate_signal_win_rate(self, res):
        '''计算信号胜率'''
        signal_win_rate = round(np.sum(np.where(res>0, 1, 0)) / len(res), 3)
        return signal_win_rate

    def caculate_total_profit_loss_ratio(self, res):
        '''计算总盈亏比'''
        return round(np.sum(res[res>0]) / np.abs(np.sum(res[res<0])), 3)

    def caculate_average_profit_loss_ratio(self, res):
        '''计算平均盈亏比'''
        return round(np.mean(res[res>0]) / np.abs(np.mean(res[res<0])), 3)

    def caculate_trade_times(self, df, res):
        '''计算交易频率'''
        return round(len(res) / len(df), 5)

    def caculate_longest_holding_period(self, holding_period):
        '''最长持仓周期'''
        return holding_period.max()

    def caculate_average_holding_period(self, holding_period):
        '''平均持仓周期'''
        return round(holding_period.mean(), 2)

    def caculate_win_loss_rate(self, df: pd.DataFrame):
        '''持仓涨跌胜率'''
        df_up = df[df['pos']>0]['profit']
        df_down = df[df['pos']<0]['profit']
        up_rate = round(np.sum(np.where(df_up>0, 1, 0)) / len(df_up), 3)
        down_rate = round(np.sum(np.where(df_down>0, 1, 0)) / len(df_down), 3)
        return up_rate, down_rate

    def caculate_total_return(self, res_rate):
        '''计算总收益'''
        return round(np.sum(res_rate), 3)

    def caculate_statistics(self, df, res, holding_period, res_rate, prefix=''):
        '''计算统计结果:
        '''
        win_rate = self.caculate_signal_win_rate(res)
        total_profit_loss_ratio = self.caculate_total_profit_loss_ratio(res)
        average_profit_loss_ratio = self.caculate_average_profit_loss_ratio(res)
        trade_times = self.caculate_trade_times(df, res)
        longest_holding_period = self.caculate_longest_holding_period(holding_period)
        average_holding_period = self.caculate_average_holding_period(holding_period)
        up_rate, down_rate = self.caculate_win_loss_rate(df)

        max_rate, min_rate, mean_rate = np.max(res_rate), np.min(res_rate), np.mean(res_rate)
        res_dic = {f'{prefix}信号胜率': [win_rate], f'{prefix}总盈亏比': [total_profit_loss_ratio],
                   f'{prefix}平均盈亏比': [average_profit_loss_ratio],
                   f'{prefix}交易频率': [trade_times], f'{prefix}最长持仓周期': [longest_holding_period],
                   f'{prefix}平均持仓周期': [average_holding_period], f'{prefix}持多头的胜率': [up_rate], f'{prefix}持空头的胜率': [down_rate],
                   f'{prefix}每笔最大盈利': [max_rate], f'{prefix}每笔最大亏损': [min_rate], f'{prefix}每笔平均盈亏': [mean_rate]}
        # res_dic = {'win_rate': win_rate, 'total_profit_loss_ratio': total_profit_loss_ratio,
        #            'average_profit_loss_ratio': average_profit_loss_ratio,
        #            'trade_times': trade_times, 'longest_holding_period': longest_holding_period,
        #            'average_holding_period': average_holding_period, 'win_loss_rate': win_loss_rate}
        return res_dic
    
    def caculate_statistics_total(self, df_li, res_save_pa='', prefix='', prefix_li = ['df_2019-05-01', 'df_2020-05-01', 'df_2020-11-01']):
        '''计算全品种统计结果:
        '''
        res_dic_li = []
        for df, prefix_i in zip(df_li, prefix_li):
            res_dic = self.caculate_statistics_single(df, suffix=prefix)
            res_dic.update({'交易频率': self.get_mean_trade_count(res_save_pa, prefix_i)})
            res_dic_li.append(res_dic)
        df_res = pd.concat([pd.DataFrame(i) for i in res_dic_li]).T
        df_res.columns = ['train', 'val', 'test']
        if len(res_save_pa) != 0:
            df_res.to_csv(f'{res_save_pa}statistic_{prefix}.csv')
        return df_res

    def get_mean_trade_count(self, pa, suffix):
        '''计算所有品种平均开仓次数'''
        pa_li = os.listdir(pa)
        df_pa_li = filter_str(suffix, pa_li, is_list=1)
        trade_rate_li = []
        for pa_i in df_pa_li:
            df = pd.read_csv(f'{pa}{pa_i}')
            trade_rate_li.append(round(len(df[df['signal']!=0]) / len(df), 5))
        return np.mean(trade_rate_li)

    def caculate_statistics_single(self, df, save_pa=None, suffix='', need_whole=0):
        '''计算单品种统计结果'''
        df.reset_index(inplace=True)
        pnl_cost_total = df[f'pnl_cost{suffix}']
        pnl_pct = pnl_cost_total.pct_change().values[1:]
        pnl_diff = pnl_cost_total.values[1:] - pnl_cost_total.values[:-1]
        total_return = pnl_cost_total.iloc[-1] - 1  # 总收益率
        sharp_ratio = np.mean(pnl_diff) / np.std(pnl_diff) * np.sqrt(244*6)  # 夏普比率

        df["highlevel"] = (
                df[f"pnl_cost{suffix}"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
        df["drawdown"] = df[f"pnl_cost{suffix}"] - df["highlevel"]
        # df["ddpercent"] = df["drawdown"] / df["highlevel"]
        max_ddpercent = df["drawdown"].min()   # 百分比最大回撤
        profit_loss_rate_all = abs(np.sum(pnl_diff[pnl_diff>0]) / np.sum(pnl_diff[pnl_diff<0]))     # 总盈亏比
        profit_loss_rate_mean = abs(np.mean(pnl_diff[pnl_diff>0]) / np.mean(pnl_diff[pnl_diff<0]))   # 平均盈亏比
        # print((str_to_datetime(df['datetime'].iloc[-1]) - str_to_datetime(df['datetime'].iloc[0])).days)
        # annual_return = self.caculate_annual_return(df, suffix=suffix)
        annual_return = total_return / ((str_to_datetime(df['datetime'].iloc[-1]) - str_to_datetime(df['datetime'].iloc[0])).days / 365)   # 年化收益率
        return_drawdown_ratio = abs(annual_return / max_ddpercent)   # 收益回撤比

        win_rate = len(pnl_diff[pnl_diff>0]) / len(pnl_diff)
        prefix = ''
        res_dic = {f'{prefix}总收益率': [total_return], f'{prefix}胜率': [win_rate], f'{prefix}总盈亏比': [profit_loss_rate_all],
                f'{prefix}平均盈亏比': [profit_loss_rate_mean], f'{prefix}收益回撤比': [return_drawdown_ratio], 
                f'{prefix}年化收益率': [annual_return], f'{prefix}夏普比率': [sharp_ratio], f'{prefix}百分比最大回撤': [max_ddpercent]
                }

        if need_whole:
            symbol = suffix[1:]
            symbol_size = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]['size'].iloc[0]
            res, holding_period, res_rate, dt = self._caculate_each_pnl(df, symbol_size)
            res_dic.update(self.caculate_statistics(df, res, holding_period, res_rate, prefix=''))
            
        if save_pa is not None:
            df_res = pd.DataFrame(res_dic).T
            df_res.to_csv(save_pa)
            return df_res
        return res_dic

    def caculate_signal_class(self, df_signal: pd.DataFrame):
        '''计算信号分类'''
        signal_class = df_signal['signal_class'].value_counts().index.to_list()
        signal_dic = {}
        for i in signal_class:
            res = df_signal[df_signal['signal_class']==i]['res'].values
            win_rate = self.caculate_signal_win_rate(res)
            total_profit_loss_ratio = self.caculate_total_profit_loss_ratio(res)
            total_profit = round(np.sum(res), 6)

            signal_dic.update({f'{i}_信号数量': [len(res)],
                               f'{i}_信号胜率': [win_rate],
                               f'{i}_总盈亏比': [total_profit_loss_ratio],
                               f'{i}_总收益率': [total_profit]})
        return signal_dic

    def caculate_statistics_all(self, train_pa=None, save_pa=None, symbol=None, need_signal=0):
        '''统计训练集验证集和测试集的结果'''
        res = []
        symbol_size = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]['size'].iloc[0]
        self.pa = self.get_train_val_test_pa(train_pa)
        for i, pa in enumerate(self.pa):
            df = pd.read_csv(f'{pa}.csv')
            if need_signal:
                res_i, holding_period, res_rate, df_signal = self._caculate_signal_pl(df, symbol_size)
                signal_dic = self.caculate_signal_class(df_signal)
            else:
                res_i, holding_period, res_rate, dt = self._caculate_each_pnl(df, symbol_size)
                signal_dic = {}

            df_res_i = pd.DataFrame({'datetime': dt, 'res': res_i, 'holding_period': holding_period, 'res_rate': res_rate,
                                     'sign_res': np.sign(res_i)})
            sp = f'{save_pa}_{self.model_li[i]}'
            df_res_i.to_csv(f'{sp}.csv', encoding='utf-8-sig', index=False)
            # m_plot_hist([df_res_i.iloc[:, i] for i in range(df_res_i.shape[1])], enumerate_li[i], sp)

            res_dic1 = self.caculate_statistics(df, res_i, holding_period, res_rate)
            res_dic = self.caculate_statistics_single(df)
            res_dic.update(res_dic1)
            res_dic.update(signal_dic)
            res.append(res_dic)
            # print(res[-1])
        
        df_res = pd.concat([pd.DataFrame(i) for i in res]).T
        df_res.columns = self.model_li
        if save_pa is not None:
            df_res.to_csv(f'{save_pa}.csv', encoding='utf-8-sig')
        return df_res

    def caculate_statistics_trend_shock_train_val_test(self, symbol, train_pa, save_pa):
        '''训练集验证集和测试集的趋势和震荡的统计结果'''
        load_pa_li = self.get_train_val_test_pa(train_pa)
        for i, pa in enumerate(load_pa_li):
            sp = f'{save_pa}_{self.model_li[i]}'
            # self.caculate_statistics_trend_shock(symbol, pa, sp)
            try:
                self.caculate_statistics_trend_shock(symbol, pa, sp)
            except:
                print(train_pa, 'got wrong.')

    def caculate_real_return(self, diff_days, total_return):
        '''按比例计算实际收益率'''
        return total_return / diff_days * 365

    def caculate_statistics_trend_shock(self, symbol, load_pa, save_pa=None):
        '''趋势和震荡的统计结果'''
        df = pd.read_csv(f'{load_pa}.csv') if isinstance(load_pa, str) else load_pa 
        symbol_size = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]['size'].iloc[0]
        df_trade_record = self._caculate_each_pnl(df, symbol_size, return_df=1)
        df_trade_record['date'] = df_trade_record['datetime'].apply(lambda x: x.split(' ')[0])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df_trend_info = get_trend_segment(symbol, df.copy(), plot=0, is_sep=self.is_sep)      # is_sep=0 
        df_trend_info['diff_day'] = pd.to_datetime(df_trend_info['trading_day_end']) - pd.to_datetime(df_trend_info['trading_day_start'])
        total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
        trend_days = df_trend_info['diff_day'].apply(lambda x: x.days).sum()
        shock_days = total_days - trend_days
        days_li = [trend_days, shock_days]
        df_trend_record_li = []
        for i in range(len(df_trend_info)):     # 将趋势和震荡信号分离
            trade_day_start, trade_day_end = df_trend_info[['trading_day_start', 'trading_day_end']].iloc[i]
            # print(type(trade_day_end), type(df_trade_record['date'].iloc[0]))
            # print(trade_day_start, trade_day_end, df_trade_record['date'].iloc[0])
            df_trend_record = df_trade_record[(df_trade_record['date']>trade_day_start) & (df_trade_record['date']<trade_day_end)]
            df_trend_record_li.append(df_trend_record.copy())
        df_trend_record = pd.concat(df_trend_record_li).dropna()
        df_shock_record = df_trade_record[~df_trade_record.index.isin(df_trend_record.index.copy())].dropna()
        # df_dic = {'趋势': df_trend_record, '震荡': df_shock_record}
        stat_res_li = []

        for i, df_i in enumerate([df_trend_record, df_shock_record]):
            '''计算趋势和震荡的统计结果'''
            res, holding_period = df_i['res'].values, df_i['holding_period'].values
            res_rate = df_i['res_rate'].values
            win_rate = self.caculate_signal_win_rate(res)
            total_profit_loss_ratio = self.caculate_total_profit_loss_ratio(res)
            average_profit_loss_ratio = self.caculate_average_profit_loss_ratio(res)
            longest_holding_period = self.caculate_longest_holding_period(holding_period)
            average_holding_period = self.caculate_average_holding_period(holding_period)
            total_return = self.caculate_total_return(res_rate)
            annual_real_return = self.caculate_real_return(days_li[i], total_return)
            
            max_rate, min_rate, mean_rate = np.max(res_rate), np.min(res_rate), np.mean(res_rate)
            res_dic = {f'总收益': [total_return], f'信号胜率': [win_rate], f'总盈亏比': [total_profit_loss_ratio],
                    f'平均盈亏比': [average_profit_loss_ratio], f'最长持仓周期': [longest_holding_period],
                    f'平均持仓周期': [average_holding_period], f'每笔最大盈利': [max_rate], 
                    f'每笔最大亏损': [min_rate], f'每笔平均盈亏': [mean_rate], f'按比例实际年化收益': annual_real_return}
            stat_res_li.append(pd.DataFrame(res_dic))
        
        df_res_concat = pd.concat(stat_res_li)
        df_res_concat.index = ['趋势', '震荡']
        df_res_concat = df_res_concat.T

        if save_pa is not None:
            df_res_concat.to_csv(f'{save_pa}.csv', encoding='utf-8-sig')

        return df_res_concat

    def get_trade_result(self, pa=f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/total_test/'):
        ''''''
        li = os.listdir(pa)
        li = filter_str('df_', li, is_list=1)
        li.remove('df_res_all.csv')
        def set_dic(x, res_dic):
            res_dic['datetime'].append(x['datetime'])
            res_dic['trade_price'].append(np.mean(eval(x['trade_price'])))
            res_dic['direction'].append(np.sign(x['pos']))
            res_dic['hand'].append(np.abs(x['pos']))

        for pa_i in li:
            print(pa_i)
            res_dic = {'datetime': [], 'trade_price': [], 'direction': [], 'hand': []}
            x_func = partial(set_dic, res_dic=res_dic)
            df = pd.read_csv(f'{pa}{pa_i}')
            df = df[(df['trade_price']!= '[]') & (df['trade_price']!= '0')]
            df.apply(x_func, axis=1)
            df_res = pd.DataFrame(res_dic)
            save_pa = pa_i.split('.')[0] + '_trade_result.csv'
            df_res.to_csv(f'{pa}{save_pa}', index=False)
            print(pa_i, 'done')
            # print(df['trade_price'].iloc[6])
            # print(type(df['trade_price'].iloc[6]))
            # input()

        

class ConcatStatistics:
    '''对统计结果进行汇总'''
    def __init__(self, pa=f'{pa_prefix}/datas/ml_result/', save_pa=f'{pa_prefix}/datas/ml_result_combo/') -> None:
        self.pa = pa
        self.symbol_li = list(filter(lambda x: len(x)<3, os.listdir(pa)))
        self.save_pa = makedir(save_pa)
        self.data_group = ['train', 'val', 'test']

    def plot_statistic_summary_symbol(self, symbol):
        '''将每个品种的模型单独汇总, 主要是图和统计结果'''
        print(symbol)
        symbol_pa = os.listdir(self.pa+symbol)
        symbol_sp = makedir(f'{self.save_pa}{symbol}/')
        symbol_plot_pa = makedir(f'{symbol_sp}{symbol}_plot/')
        symbol_trend_shock_pa = makedir(f'{self.save_pa}trend_shock_statistic/')
        df_li = []
        df_trend_shock_li = []
        for pa_i in symbol_pa:
            folder_pa = f'{self.pa}{symbol}/{pa_i}/'
            folder_li = os.listdir(folder_pa)
            plot_pa = filter_str('.png', folder_li) 
            shutil.copy(f'{folder_pa}{plot_pa}', symbol_plot_pa)
            try:
                statis_pa = filter_str('statistics.csv', folder_li)
                df_trend_shock, _ = self.trend_shock_summary(folder_pa, folder_li)
            except:
                continue
            df_i = pd.read_csv(folder_pa + statis_pa)
            df_i.columns = [statis_pa, 'train', 'val', 'test']
            df_i[''] = None
            df_li.append(df_i.copy())

            df_trend_shock_li.append(df_trend_shock.copy())

        if len(df_trend_shock_li) != 0 and len(df_li) != 0:
            pd.concat(df_li, axis=1).to_csv(f'{symbol_sp}statis_{symbol}.csv', index=False)
            pd.concat(df_trend_shock_li, axis=1).to_csv(f'{symbol_sp}statis_trend_shock_{symbol}.csv', index=False)

    def run_all_symbols(self, func):
        '''汇总所有品种的统计结果'''
        for symbol in self.symbol_li:
            func(symbol)

    def trend_shock_summary(self, folder_pa, folder_li):
        '''趋势和震荡部分结果统计'''
        suffix = 'shock_statistic_'
        df_li = []
        col_i = None
        rate_dic = {"name": [], "train_trend": [], "train_shock": [], "val_trend": [], 
                    "val_shock": [], "test_trend": [], "test_shock": [], "train_score": [], "val_score": [], "test_score": [], "train_val_score": [],
                    "train_rate": [], "val_rate": [], "test_rate": []}
        for i in self.data_group:
            pa_i = filter_str(suffix+i, folder_li) 
            df = pd.read_csv(folder_pa+pa_i)
            if col_i is None:
                col_i = pa_i
            df.columns = [col_i, "趋势", "震荡"]
            df.loc[len(df)] = None
            df[''] = None
            df_li.append(df.copy())
            trend_value, shock_value = df["趋势"].iloc[-2], df["震荡"].iloc[-2]

            rate_dic[f"{i}_trend"].append(trend_value)
            rate_dic[f"{i}_shock"].append(shock_value)

            if i == 'train':
                is_trend = 1 if trend_value > shock_value else 0
            # if (trend_value <= 0 and shock_value <= 0) or (is_trend == 1 and i == "val" and trend_value < shock_value) or (is_trend == 0 and i == "val" and trend_value > shock_value):
            #     rate_dic[f"{i}_score"].append(-10)
            # else:
            if is_trend:
                score = (trend_value+shock_value)*(trend_value-shock_value)
                trend_shock_rate = abs(trend_value / shock_value)
            else:
                score = (trend_value+shock_value)*(shock_value-trend_value)
                trend_shock_rate = abs(shock_value / trend_value) 

            rate_dic[f"{i}_score"].append(score)
            # 2022.10.31 对分数比值加权求和-两两比值相差的绝对值
            rate_dic[f"{i}_rate"].append(trend_shock_rate)

        rate_dic[f"train_val_score"].append(rate_dic[f"train_score"][0]+rate_dic[f"val_score"][0])

        rate_dic["name"].append(pa_i)
        df_res = pd.concat(df_li)

        return df_res, rate_dic

    def caculate_annual_return(self, df, suffix=''):
        '''计算年化收益率'''
        total_return = df[f'pnl_cost{suffix}'].iloc[-1] - 1  # 总收益率
        annual_return = total_return / ((df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days / 365)   # 年化收益率
        return annual_return
        
    def caculate_holding_rate(self, pa=None, is_tvt=0, df0=None):
        '''计算多空持仓比例, 读取analyze文件'''
        def func(pa, df0):
            if df0 is None: df_o = pd.read_csv(pa)
            else: df_o = df0.copy()
            df_o.reset_index(inplace=True)
            df = df_o[df_o['signal']!=0].copy()
            if len(df):
                df['ind'] = df.index
                df['re'] = df['ind'].shift(-1) - df['ind']
                df['re'].iloc[-1] = len(df_o) - df['ind'].iloc[-1]
                holding_rate = df[df['signal']==1]['re'].sum() / len(df_o)
            else:
                holding_rate = 0
            return holding_rate
        if is_tvt:
            train_pa, val_pa, test_pa = pa, pa.replace('train', 'val'), pa.replace('train', 'test')
            return func(train_pa, df0), func(val_pa, df0), func(test_pa, df0)
        else:
            return func(pa, df0)

    def trend_shock_score(self, symbol):
        '''将每个品种的模型单独汇总, 主要是图和统计结果'''
        print(symbol)
        method = 0
        symbol_pa = os.listdir(self.pa+symbol)
        total_statistic_pa = makedir(f'{self.save_pa}trend_shock_statistic/total_statistic/')
        annual_rate_pa = makedir(f'{self.save_pa}trend_shock_statistic/annual_rate/')
        df_trend_shock_li = []
        rate_res = []
        ms = ModelStatistics()  # 统计回测结果
        ld_pa = f'{self.pa}{symbol}/'
        for pa_i in os.listdir(ld_pa):
            train_pa = filter_str('train_analyze.csv', os.listdir(f'{ld_pa}{pa_i}'))
            train_pa = f'{ld_pa}{pa_i}/{train_pa}'[:-4]
            save_pa = train_pa.replace('_train_analyze', '')
            # save_pa = f'{ld_pa}{pa_i}/{save_pa}'
            ms.caculate_statistics_trend_shock_train_val_test(symbol, train_pa, f'{save_pa}_trend_shock_statistic')

        for pa_i in symbol_pa:
            folder_pa = f'{self.pa}{symbol}/{pa_i}/'
            folder_li = os.listdir(folder_pa)
            analyze_pa = f'{self.pa}{symbol}/{pa_i}/' + filter_str('_train_analyze.csv', folder_li)
            try:
                df_trend_shock, rate_dic = self.trend_shock_summary(folder_pa, folder_li)
            except:
                # print(pa_i, '11111111111111')
                continue
            
            if method == 0:
                rate_res.append(rate_dic)
            elif method == 1:
                holding_rate = self.caculate_holding_rate(analyze_pa, is_tvt=1)       # 判断多空持仓周期是否严重不平衡
                max_holding_rate, min_hodling_rate = np.max(holding_rate), np.min(holding_rate)
                # print(max_holding_rate, min_hodling_rate)
                if max_holding_rate < 0.8 and min_hodling_rate > 0.2:
                    rate_res.append(rate_dic)
            elif method == 2:
                holding_rate = self.caculate_holding_rate(analyze_pa, is_tvt=1)       # 判断多空持仓周期是否严重不平衡
                max_holding_rate, min_hodling_rate = np.max(holding_rate), np.min(holding_rate)
                # print(max_holding_rate, min_hodling_rate)
                if max_holding_rate < 0.7 and min_hodling_rate > 0.3:
                    rate_res.append(rate_dic)
            df_trend_shock_li.append(df_trend_shock.copy())

        if len(rate_res) != 0:
            pd.concat(df_trend_shock_li, axis=1).to_csv(f'{total_statistic_pa}statis_trend_shock_{symbol}.csv', index=False)
            df_res = pd.concat([pd.DataFrame(i) for i in rate_res])
            
            df_adj = df_res[((df_res["train_trend"]>df_res["train_shock"]) & (df_res["val_trend"] > df_res["val_shock"]) & \
                (df_res["test_trend"] > df_res["test_shock"])) | 
                ((df_res["train_trend"]<df_res["train_shock"]) & (df_res["val_trend"] < df_res["val_shock"]) & \
                    (df_res["test_trend"]<df_res["test_shock"]))]
            df_adj = df_res[((df_res["train_trend"]+df_res["train_shock"]>0) & (df_res["val_trend"] + df_res["val_shock"]>0) & \
                (df_res["test_trend"] + df_res["test_shock"]>0))]
            df_adj = df_adj[(df_adj["train_score"]>0) & ((df_adj["train_trend"]>0) | (df_adj["train_shock"]>0))]
            df_adj = df_adj[(df_adj["val_score"]>0) & ((df_adj["val_trend"]>0) | (df_adj["val_shock"]>0))]
            df_adj = df_adj[df_adj['test_score']>0]
            df_res = df_adj[((df_adj["test_trend"]>-0.04) & (df_adj["test_shock"]>-0.04)) | ((df_adj["test_trend"]>0) | (df_adj["test_shock"]>0))]

            # df_res = df_res[(df_res['train_trend']<1.0) & (df_res['train_shock']<1.0)] # 1.0没有
            
            # 2022.10.31 对分数比值加权求和-两两比值相差的绝对值
            df_res['score1'] = df_res['train_rate']*0.2 + df_res['val_rate']*0.3 + df_res['test_rate']*0.5 # - \
                # (df_res['train_rate'] - df_res['val_rate']).abs() - (df_res['val_rate'] - df_res['test_rate']).abs() - \
                # (df_res['test_rate'] - df_res['train_rate']).abs()
            
            df_res["train_rank"] = df_res["train_score"].rank()
            df_res["val_rank"] = df_res["val_score"].rank()
            df_res["train_val_rank"] = df_res["train_rank"]+df_res["val_rank"]
            df_res.sort_values('train_val_score').to_csv(f'{annual_rate_pa}rate_trend_shock_{symbol}.csv', index=False)
        return 0

    def concat_table(self, pa=None, method=0):
        if pa is None:
            pa = f'{self.save_pa}trend_shock_statistic/annual_rate/'
        pa_li = os.listdir(pa)
        save_pa = f'{self.save_pa}trend_shock_statistic/'
        df_li = []
        for i in pa_li:
            df_i = pd.read_csv(pa+i)
            symbol = i.split('_')[-1].split('.')[0]
            df_i['symbol'] = symbol
            is_select_li = []
            for j in range(len(df_i)):  # 获取年化收益，把收益太高或者小于-0.04的去掉
                # pa_j = df_i['name'].iloc[j].replace('_trend_shock_statistic_test.csv', '').replace('y_pred_', '')
                # # 要改 1.0版本没有年化收益率
                # df_j = pd.read_csv(f'{self.pa}{symbol}/{pa_j}/y_pred_{pa_j}_statistics.csv')
                # df_j.columns = ['name', 'train', 'val', 'test']
                # df_j = df_j[df_j['name']=='年化收益率']
                # max_rate, min_rate = df_j.max(axis=1).iloc[0], df_j.min(axis=1).iloc[0]
                # is_select = 0 if (min_rate < -0.01) else 1    # 过滤过拟合和欠拟合的模型
                is_select = 1
                is_select_li.append(is_select)

            df_i['select'] = is_select_li
            df_i = df_i[df_i['select']==1]

            if method == 0:
                df_i = df_i[(df_i['test_score'] == df_i['test_score'].max())]   # score0
            elif method == 1:
                df_i = df_i[(df_i['score1'] == df_i['score1'].max())]   # score1
            elif method == 2:
                if len(df_i) > 1:
                    df_i = df_i.sort_values('test_score', ascending=False, ignore_index=True)
                    df_i = df_i[df_i.index == 1]
                else:
                    df_i = df_i[(df_i['test_score'] == df_i['test_score'].max())]

            df_li.append(df_i)
        df = pd.concat(df_li)
        df = df[(df['test_score']>0)&(df['train_val_score']>0)]
        df.to_csv(save_pa+'table_total.csv', index=False)

        # load_pa = f'{pa_prefix}/datas/ml_result/'
        for i in range(len(df)):
            pa_name = df['name'].iloc[i]
            pa_name_li = pa_name.split('_')
            symbol, y_thread, factor_thread = pa_name_li[3], pa_name_li[2], pa_name_li[5]
            sy_pa_li = os.listdir(f'{self.pa}{symbol}')
            sy_pa = filter_str(factor_thread, filter_str(y_thread, sy_pa_li, is_list=1))
            shutil.copytree(f'{self.pa}{symbol}/{sy_pa}', f'{self.save_pa}model_2.0/{sy_pa}')
        return df

    def concat_total_statistic(self, pa: str):
        '''把计算出来的统计结果汇总，用于筛选模型后的结果统计'''
        pa_li = os.listdir(pa)
        model_name = pa.split('/')[-2]
        pa_li.remove('params')
        df_li = []
        for pa_i in pa_li:
            df_i = pd.read_csv(f'{pa}{pa_i}/total_train/statistic__total.csv')
            df_i.columns = ['index', f'{pa_i}_train', f'{pa_i}_val', f'{pa_i}_test']
            df_i.set_index('index', inplace=True)
            df_li.append(df_i.T)

        df_concat = pd.concat(df_li)

        df_concat.to_csv(f'{pa}{model_name}_df_res.csv')
        return df_concat


# pa_prefix = 'D:/策略开发/futures_ml'

def run_modelstatistics():
    '''模型结果统计'''
    # train_pa = 'y_pred_[5, 0.5, 1, 1]_v_60m_1.2_20_1_return_rate_60m_train_analyze'
    # train_pa = f'{pa_prefix}/simulation/optuna_params/pp/y_pred_[10, 0.5, 1, 1]_pp_60m_1.3_15_1_return_rate_60m_train_analyze'
    # save_pa = f'{pa_prefix}/simulation/optuna_params/pp/y_pred_[10, 0.5, 1, 1]_pp_60m_1.3_15_1_return_rate_60m_modelstatistics'
    symbol = 'SN'
    # train_pa = f'{pa_prefix}/simulation/optuna_params/HC/y_pred_[7, 0.6, 1, 1]_HC_60m_1.3_sample_16_1_return_rate_60m_train_analyze'
    # save_pa = f'{pa_prefix}/simulation/optuna_params/HC/y_pred_[7, 0.6, 1, 1]_HC_60m_1.3_sample_16_1_return_rate_60m_modelstatistics'
    # train_pa = f'{pa_prefix}/filter_results/v/res7/y_pred_[7, 0.6, 1, 1]_v_60m_1.2_sample_20_1_return_rate_60m_train_analyze'
    # save_pa = f'{pa_prefix}/filter_results/v/res7/y_pred_[7, 0.6, 1, 1]_v_60m_1.2_sample_20_1_return_rate_60m_modelstatistics'
    train_pa = f'{pa_prefix}/datas/ml_result/total/[10, 1, 1, 1]_SN_60m_1.3_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_SN_60m_1.3_sample_10_1_return_rate_60m_train_analyze'
    save_pa = f'{pa_prefix}/datas/ml_result/total/[10, 1, 1, 1]_SN_60m_1.3_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_SN_60m_1.3_sample_10_1_return_rate_60m_statistics'
    ms = ModelStatistics()
    ms.caculate_statistics_all(train_pa=train_pa, save_pa=save_pa, symbol=symbol)
    # ms.caculate_statistics_total()
    print('run_modelstatistics done.')

def run_statistics_trend_shock():
    '''趋势和震荡结果统计'''
    symbol = 'JD'
    # load_pa = f'{pa_prefix}/datas/backtest_res/RB/y_pred_[5, 0.5, 1, 1]_RB_60m_1.2_sample_10_1_return_rate_60m_train_analyze'
    load_pa = f'{pa_prefix}/datas/ml_result/symbol_result_adj/params/[5, 0.5, 1, 1]_JD_60m_1.3_sample_20_1_return_rate_60m/y_pred_[5, 0.5, 1, 1]_JD_60m_1.3_sample_20_1_return_rate_60m_test_analyze'
    save_pa = 'trend_shock'
    ms = ModelStatistics()
    ms.caculate_statistics_trend_shock(symbol, load_pa, save_pa)

def run_concatstatistics(pa='', save_pa=''):
    # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std/model_raw'
    if len(pa) == 0:
        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_early_stop'
        # pa = f'{pa_prefix}/datas/ml_result/model_1.0'
        pa = f'{pa_prefix}/datas/ml_result/model_2.0/original'
        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_3'
        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/original_max_ddpercent_2'
        pa = f'{pa_prefix}/datas/ml_result/model_2.0/adj_target_original_10_15'
        # save_pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_early_stop/model_zigzag'
        # save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_3/model_raw')
        save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_2.0/adj_target_original_10_15/model_raw')
        # save_pa = f'{pa_prefix}/datas/ml_result/model_2.0/original/model_raw'
        # save_pa = f'{pa_prefix}/datas/ml_result/model_1.0/model_raw'
        
    cs = ConcatStatistics(f'{pa}/params/', f'{save_pa}/')
    cs.run_all_symbols(cs.trend_shock_score)
    # cs.run_all_symbols(cs.plot_statistic_summary_symbol)
    cs.concat_table(method=0)

def run_concat_total_statistic():
    pa = f'{pa_prefix}/datas/ml_result/model_2.0/skew_add_reduce_10_7/'
    save_pa = f'{pa_prefix}/datas/ml_result/model_2.0/skew_add_reduce_10_7/model_raw'
    cs = ConcatStatistics(f'{pa}/params/', f'{save_pa}/')
    cs.concat_total_statistic(pa)



if __name__ == '__main__':
    run_concatstatistics()
    # ms = ModelStatistics()
    # ms.get_trade_result()

    # run_concat_total_statistic()

    
            
