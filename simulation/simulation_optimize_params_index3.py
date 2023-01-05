from cgi import test
from cmath import nan
from doctest import FAIL_FAST
import enum
import imp
from multiprocessing.spawn import prepare
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
from datas_process.m_datas_process import run_nkl
# from tkinter.messagebox import NO
from matplotlib.pyplot import step
import matplotlib
import optuna
import joblib
from datetime import datetime
from machine_learning.trainmodel import TrainClassification, TrainClassificationCV, run_train
from m_base import makedir, save_json
from datas_process.m_datas_process import run_dp
import pandas as pd
__Author__ = 'ZCXY' 
import numpy as np
from simulation.simulation_backtester import SimulationBackTester
from simulation.simulation_strategy import SimulationStrategy
from simulation.simulation_strategy_index import SimulationIndexStrategy
from simulation.base_strategy import BaseStrategy
from backtesting.ml_strategy_adj import MLStrategy
import shutil
from backtesting.model_statistics import ModelStatistics
from machine_test.ml_test import *
import matplotlib.pyplot as plt
from backtesting.data_analyze_show import plot_pnl_seperate, plot_show1, plot_show_index_res
from datas_process.m_futures_factors import SymbolsInfo

# matplotlib.use('Agg')


def init_val():
    '''初始化变量'''
    pa = f'{pa_prefix}/datas/ml_result/symbol_result_adj/params/'
    suffix_li = os.listdir(pa)
    symbol_li = [i.split('_')[1] for i in suffix_li]
    y_pred_li = [f'{pa}{i}/y_pred_{i}.csv' for i in suffix_li]
    res_pa = f'{pa_prefix}/simulation/optuna_params/'
    ms = ModelStatistics()
    return symbol_li, y_pred_li, res_pa, suffix_li, ms

def caculate_statistics(res_pa, symbol, ms, need_signal=0, suffix=''):
    '''计算统计指标'''
    st_pa = f'{res_pa}{symbol}/'
    res_pa_li = os.listdir(st_pa)
    train_pa = list(filter(lambda x: 'train_analyze' in x, res_pa_li))[0]
    train_pa = f'{st_pa}{train_pa}'[:-4]
    save_pa = train_pa.replace('train_analyze', 'modelstatistics')
    df_statis = ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_{suffix}', symbol=symbol, need_signal=need_signal)
    return df_statis

def run_optimize_total_mltest():
    '''全品种优化'''
    # symbol_li, y_pred_li, res_pa, suffix_li, ms = init_val()
    symbol = 'total'
    n_trials = 17
    suffix = 'total'
    res_save_pa = makedir(f'{pa_prefix}/simulation/optuna_params/{symbol}/')
    symbol_li = SymbolsInfo().symbol_li
    sbt = SimulationBackTester(strategy_class=SimulationIndexStrategy)
    target_type = 'max_ddpercent'
    params = {"trade_type": 0, "up_limit": 70, "down_limit": 30, "index_name": "rsi", "index_n": 9, "signal_type": 0}

    print(params)

    _, df_res_all = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2016, 1, 1), enddate=datetime(2019, 8, 1), 
                                            y_pred_li=None, params=params, delay=20, target_type='drawdown', save_pa=res_save_pa)
    df_res_all.reset_index(drop=True, inplace=True)
    df_res_all.set_index('datetime', inplace=True)
    # df_res_all.to_csv(f'{res_save_pa}df_res_all.csv')
    df = pd.DataFrame(df_res_all['pnl_cost_total'])
    # df.plot()
    # plt.show()
    # df_res_all = pd.read_csv(f'{res_save_pa}df_res_all.csv')
    print(df_res_all.tail(10))
    plot_pnl_seperate(df_res_all, res_save_pa)

    ms = ModelStatistics()
    ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{symbol}.csv', suffix=f'_{symbol}')

    symbol_analyze_li = os.listdir(f'{res_save_pa}')
    for sy in symbol_li:
        print(sy)
        ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy}.csv', suffix=f'_{sy}')
        pa = res_save_pa + filter_str(f'test_{sy}', symbol_analyze_li)
        plot_show1(sy, pa=pa, save_pa=f'{res_save_pa}analyze_{sy}.jpg', mod=0)
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy2}.csv', suffix=f'_{sy2}')
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy3}.csv', suffix=f'_{sy3}')
    print('done.')

    '''检验交易结果是否正确'''
    # df_res, _ = sbt.all_contract_backtesting('AP', startdate=datetime(2020, 5, 1), enddate=datetime(2020, 11, 1), y_pred=y_pred_li[0], 
    #                             params=params, delay=0, target_type=target_type)
    # df_res.to_csv('df_res.csv')
    return 

def get_gplearn_index(pa='good', my_pa=None, replace_str='.npy'):
    if my_pa is not None:
        load_pa = my_pa
    else:
        load_pa = f'{pa_sys}search_factor/my_gplearn/{pa}/'
    index_name_li = os.listdir(load_pa)
    index_name_li = [i.replace(replace_str, '') for i in index_name_li]
    return index_name_li

def index_analyze(index_dic={}, symbol='', pa_name='raw_index_result'):
    '''指标回测分析'''
    # symbol_li, y_pred_li, res_pa, suffix_li, ms = init_val()

    symbol = 'total'
    n_trials = 17
    suffix = symbol
    res_save_pa = makedir(f'{pa_prefix}/simulation/optuna_params/{pa_name}/{index_dic["index_name"]}/')
    symbol_li = SymbolsInfo().symbol_li
    # symbol_li.remove('C')
    # symbol_li = ['RB', 'BU', 'RU', 'PP']
    sbt = SimulationBackTester(strategy_class=SimulationIndexStrategy)
    target_type = 'max_ddpercent'
    params = {"trade_type": 0, "up_limit": 70, "down_limit": 30, "index_name": "rsi", "index_n": 9, "signal_type": 0}
    params.update(index_dic)

    print(params)

    _, df_res_all = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2016, 1, 1), enddate=datetime(2019, 8, 1), 
                                            y_pred_li=None, params=params, delay=20, target_type='drawdown', save_pa=res_save_pa)
    df_res_all.reset_index(drop=True, inplace=True)
    df_res_all.set_index('datetime', inplace=True)
    print(df_res_all.tail(10))
    plot_pnl_seperate(df_res_all, res_save_pa)

    ms = ModelStatistics()
    ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{symbol}.csv', suffix=f'_{symbol}')

    symbol_analyze_li = os.listdir(f'{res_save_pa}')
    for sy in symbol_li:
        print(sy)
        ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy}.csv', suffix=f'_{sy}')
        pa = res_save_pa + filter_str(f'test_{sy}', symbol_analyze_li)
        plot_show_index_res(sy, pa=pa, save_pa=f'{res_save_pa}analyze_{sy}.jpg', mod=4)
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy2}.csv', suffix=f'_{sy2}')
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy3}.csv', suffix=f'_{sy3}')
    print('done.')

def multi_index_analyze():
    pa = 'good_0.003rate'
    pa = 'good_total_return_all_quantile_normal_2'
    index_name_li = get_gplearn_index(pa)
    # params = {"trade_type": 0, "signal_type": 0}
    # params = {"trade_type": 0, "up_limit": 70, "down_limit": 30, "index_name": "rsi", "index_n": 9, "signal_type": 0}
    # index_dic = [{"index_name": "rsi", "up_limit": 75, "down_limit": 25, "index_n": 9}, 
    #              {"index_name": "cci", "up_limit": 100, "down_limit": -100, "index_n": 46}, 
    #              {"index_name": "cmo", "up_limit": 50, "down_limit": -50, "index_n": 15}]
    # # for dic in index_dic:
    # #     index_analyze(dic)
    # params = {"trade_type": 0, "up_limit": 70, "down_limit": 30, "index_name": "sma", "index_n": 6, "index_n1": 12, "signal_type": 2}
    # params = {"trade_type": 0, "up_limit": 10, "down_limit": -10, "up_limit1": 5, "down_limit1": -5, 
    #           "index_name": "madifrsi", "index_n": 6, "index_n1": 12, "index_n2": 5, "signal_type": 4, "trade_type": 1}
    # params = {"index_name": "ts_decay_linear(ts_decay_linear(close, 4), 4)", "trade_type": 1, "signal_type": 5}
    # params = {"index_name": "ts_cci(turnover, open, turnover, 4)", "trade_type": 1, "signal_type": 5}
    # params = {"index_name": "ts_argmin(low, 6)", "trade_type": 1, "signal_type": 6}
    
    print(len(index_name_li))
    for index_name in index_name_li[12:16]:
        params = {"index_name": index_name, "trade_type": 1, "signal_type": 9}
        index_analyze(params, pa_name=pa)

def fitness_plot():
    '''适应度画图'''
    pa = 'good'
    is_plot = 1
    need_normalize = 2
    # pa_prefix = pa_sys[:-1]
    method = 1  # 0全品种计算累计收益率，1单品种计算累计收益率
    my_pa = f'{pa_prefix}/simulation/optuna_params/gplearn_selescted_index_best/'
    # my_pa = f'{pa_prefix}/simulation/factor_plot_adj/' # .png
    # my_pa = f'{pa_prefix}/search_factor/my_gplearn/good_cumsum_return_rate_raw/'
    my_pa = f'{pa_prefix}/search_factor/my_gplearn/good_cumsum_return_rate_adj/'
    my_pa = f'{pa_prefix}/search_factor/my_gplearn/good_total_return_all_quantile_normal_2/'
    load_contract_count_pa = f'{pa_prefix}/search_factor/my_gplearn/raw_data/df_contract_count.csv'
    save_pa = makedir(f'{pa_prefix}/simulation/factor_plot/')
    index_name_li = get_gplearn_index(pa, my_pa, replace_str='.npy')
    startdate, enddate = datetime(2016, 1, 1), datetime(2019, 5, 1)
    symbol_li = SymbolsInfo().symbol_li
    mainconinfo = MainconInfo()
    df_contract_count = pd.read_csv(load_contract_count_pa)
    df_res = pd.DataFrame({'symbol': df_contract_count.columns.to_list(), 'contract_count': df_contract_count.iloc[0].to_list()})
    k_line_li = []
    symbol_dic = {}
    # input()
    for symbol in symbol_li:
        sy_k_line_li = mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=20, load_pa=None, is_concat=0, contract_name=0)
        symbol_dic[symbol] = sy_k_line_li
        k_line_li = k_line_li + sy_k_line_li

    gpl = GPLearnIndex()
    # index_s, index_v = gpl.get_index(self.index_name, '_total_return_all_quantile')
    # index_name_li = ['1.0101_0.9997554389274561_1.0003604627314862__total_return_all_quantile__ts_wma(division(open, shift(close)), 12)']
    # index_name_li = ['1.0101_0.9997554389274561_1.0003604627314862__total_return_all_quantile__ts_wma(open, 12)']
    # index_name_li = ['0.1536__cumsum_return_rate__typprice(ts_rsi(close, 12), ts_max_diff(high, 6), ts_median(high, 12))']

    if method == 0:
        for index_name in index_name_li:  # index_name_li:
            res_li = []
            for df in k_line_li:
                df_i = df.copy()
                df_i['pre_close'] = df_i['close'].shift(1)
                df_i['open'] = df_i['open'] / df_i['pre_close']
                df_i['high'] = df_i['high'] / df_i['pre_close']
                df_i['low'] = df_i['low'] / df_i['pre_close']
                df_i['close'] = df_i['close'] / df_i['pre_close']
                df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)
                gpl.set_datas(df_i)
                _, index_v = gpl.get_index(index_name)
                df_i['return_rate'] = df_i['close'].pct_change().shift(-1)
                df_i['index_v'] = index_v 
                df_i.dropna(inplace=True)
                res_li.append(df_i[['index_v', 'return_rate']].copy())
            df_concat = pd.concat(res_li)
            df_concat.to_csv(f'{save_pa}df_{index_name}.csv', index=False)
            # print(df_concat)
            # input()
            df_concat = df_concat.sort_values('index_v', ascending=True)
            # print(df_concat['index_v'])
            fig = plt.figure(figsize=(18, 12))
            df_concat['return_rate'] = df_concat['return_rate'].cumsum()
            plt.plot(df_concat['index_v'], df_concat['return_rate'])
            plt.savefig(f'{save_pa}{index_name}.png')
            plt.close()
            print(index_name, 'done.')
    elif method == 1:
        for num, index_name in enumerate(index_name_li[:]):  # index_name_li: 22
            fitness_li = []
            df_concat_li = []
            save_index_pa = makedir(f'{pa_prefix}/simulation/factor_good_total_return_all_quantile_normal_2/{index_name}/')
            print(index_name, num, 'start.')
            # try:
            for sy, sy_k_line_li in symbol_dic.items():  # symbol_dic.items()
                res_li = []
                close_start, volume_start = 0, 0
                for n, df in enumerate(sy_k_line_li):
                    df_i = df.copy()
                    df_i.set_index('datetime', inplace=True)
                    df_i['return_rate'] = df_i['close'].pct_change().shift(-1)
                    if close_start == 0: 
                        close_start = df_i['close'].iloc[0]
                        volume_start = df_i['volume'].iloc[0]

                    if need_normalize == 1:
                        df_i['pre_close'] = df_i['close'].shift(1)
                        df_i['open'] = df_i['open'] / df_i['pre_close']
                        df_i['high'] = df_i['high'] / df_i['pre_close']
                        df_i['low'] = df_i['low'] / df_i['pre_close']
                        df_i['close'] = df_i['close'] / df_i['pre_close']
                        df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)
                    
                    elif need_normalize == 2:
                        df_i['open'] = df_i['open'] / df_i['open'].shift(1)
                        df_i['high'] = df_i['high'] / df_i['high'].shift(1)
                        df_i['low'] = df_i['low'] / df_i['low'].shift(1)
                        df_i['close'] = df_i['close'] / df_i['close'].shift(1)
                        df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)

                    # df_i['return_rate'] = df_i['close'].pct_change().shift(-1)
                    gpl.set_datas(df_i)

                    index_v = gpl.get_index(index_name, just_index_v=1)
                    index_v.index = df_i.index
                    df_i['index_v'] = index_v 
                    df_i.dropna(inplace=True)

                    # res_li.append(df_i[['index_v', 'return_rate']].copy())
                    res_li.append(df_i.copy())
                
                df_concat = pd.concat(res_li)
                # df_concat.to_csv(f'{sy}_df_concat.csv')
                # print('df_concat done.')
                # input()
                # save_index_pa = makedir(f'{pa_prefix}/simulation/factor_sep_real/{index_name}/')
                # df_concat.to_csv(f'{save_index_pa}{sy}.csv')
                df_concat = df_concat.sort_values('index_v', ascending=True)
                # df_concat.reset_index(drop=True, inplace=True)
                df_concat['cumsum_return_rate'] = df_concat['return_rate'].cumsum()
                if is_plot:
                    plt.figure(figsize=(18, 12))
                    plt.plot(df_concat['index_v'], df_concat['cumsum_return_rate'])
                    # save_index_pa = makedir(f'{pa_prefix}/simulation/factor_sep_best1/{index_name}/')
                    plt.savefig(f'{save_index_pa}{sy}.png')
                    plt.close()
                    # plt.clf()
                df_concat_li.append(df_concat)
                fitness_li.append(df_concat['cumsum_return_rate'].abs().max()-abs(df_concat['cumsum_return_rate'].iloc[-1])/2)
                # print(index_name, sy, fitness_li[-1], 'done.')
            df_concat_all = pd.concat(df_concat_li, ignore_index=True)
            df_concat_all = df_concat_all.sort_values('index_v', ascending=True)
            df_concat_all['cumsum_return_rate'] = df_concat_all['return_rate'].cumsum()
            
            if is_plot == 0:
                df_res[index_name] = fitness_li
            else:
                plt.figure(figsize=(18, 12))
                plt.plot(df_concat_all['index_v'], df_concat_all['cumsum_return_rate'])
                # save_index_pa = makedir(f'{pa_prefix}/simulation/factor_sep_best1/{index_name}/')
                plt.savefig(f'{save_index_pa}total.png')
                plt.close()
            print(index_name, num, 'done.')
            # except:
            #     print(index_name, num, 'not done.')
        if is_plot == 0:
            mean_fitness_li = []
            for i in range(len(df_res)):
                mean_fitness_li.append(df_res.iloc[i, 2:].mean())
            df_res['mean_fitness'] = mean_fitness_li
            df_res.to_csv(f'{pa_prefix}/search_factor/my_gplearn/raw_data/df_fitness_info1.csv', index=False)
            print('df_res done.')

def diff_df():
    sy = 'AG'
    index_name1 = '1.0101_0.9997554389274561_1.0003604627314862__total_return_all_quantile__ts_wma(division(open, shift(close)), 12)'
    index_name2 = '1.0101_0.9997554389274561_1.0003604627314862__total_return_all_quantile__ts_wma(open, 12)'
    pa1 = f'{pa_prefix}/simulation/factor_sep_real/{index_name1}/{sy}.csv'
    pa2 = f'{pa_prefix}/simulation/factor_sep_real/{index_name2}/{sy}.csv'
    df1 = pd.read_csv(pa1)
    df2 = pd.read_csv(pa2)
    for i, df_i in enumerate([df1, df2]):
        df_i['cumsum_return_rate'] = df_i['return_rate'].cumsum()
        plt.figure(figsize=(18, 12))
        plt.plot(df_i['index_v'], df_i['cumsum_return_rate'])
        save_index_pa = makedir(f'{pa_prefix}/simulation/factor_sep_real/{index_name1}/')
        plt.savefig(f'{sy}{i}.png')
        # plt.close()
        plt.clf()
    df = df1.copy()
    df['index_v'] = df1['index_v'] - df2['index_v']
    df['return_rate'] = df1['return_rate'] - df2['return_rate']
    df.to_csv('df.csv', index=False)

def fitness_plot1():
    '''适应度画图'''
    pa = 'good'
    is_plot = 1
    need_normalize = 1
    # pa_prefix = pa_sys[:-1]
    method = 1  # 0全品种计算累计收益率，1单品种计算累计收益率
    my_pa = f'{pa_prefix}/simulation/optuna_params/gplearn_selescted_index_best/'
    # my_pa = f'{pa_prefix}/simulation/factor_plot_adj/' # .png
    # my_pa = f'{pa_prefix}/search_factor/my_gplearn/good_cumsum_return_rate_raw/'
    load_contract_count_pa = f'{pa_prefix}/search_factor/my_gplearn/raw_data/df_contract_count.csv'
    save_pa = makedir(f'{pa_prefix}/simulation/factor_plot/')
    index_name_li = get_gplearn_index(pa, my_pa, replace_str='.npy')
    startdate, enddate = datetime(2016, 1, 1), datetime(2019, 5, 1)
    symbol_li = SymbolsInfo().symbol_li
    mainconinfo = MainconInfo()
    df_contract_count = pd.read_csv(load_contract_count_pa)
    df_res = pd.DataFrame({'symbol': df_contract_count.columns.to_list(), 'contract_count': df_contract_count.iloc[0].to_list()})
    k_line_li = []
    symbol_dic = {}
    # input()
    for symbol in symbol_li[:1]:
        sy_k_line_li = mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=20, load_pa=None, is_concat=0, contract_name=0)
        symbol_dic[symbol] = sy_k_line_li
        k_line_li = k_line_li + sy_k_line_li

    gpl = GPLearnIndex()
    # index_s, index_v = gpl.get_index(self.index_name, '_total_return_all_quantile')
    index_name_li = ['1.0101_0.9997554389274561_1.0003604627314862__total_return_all_quantile__ts_wma(division(open, shift(close)), 12)']
    index_name_li = ['1.0101_0.9997554389274561_1.0003604627314862__total_return_all_quantile__ts_wma(open, 12)']

    if method == 1:
        for num, index_name in enumerate(index_name_li[:]):  # index_name_li:
            fitness_li = []
            # try:
            for sy, sy_k_line_li in symbol_dic.items():
                res_li = []
                close_start = 0
                for n, df in enumerate(sy_k_line_li):
                    df_i = df.copy()
                    if close_start == 0:
                        close_start = df_i['close'].iloc[0]
                    df_i.set_index('datetime', inplace=True)
                    df_i['return_rate0'] = df_i['close'].pct_change().shift(-1)
                    df_i['return_rate1'] = df_i['close'].shift(-1)/df_i['close']
                    
                    if need_normalize:
                        # df_i['pre_close'] = df_i['close'].shift(1)
                        # df_i['open_rate'] = df_i['open'] / df_i['pre_close']
                        # df_i['high_rate'] = df_i['high'] / df_i['pre_close']
                        # df_i['low_rate'] = df_i['low'] / df_i['pre_close']
                        # df_i['close_rate'] = df_i['close'] / df_i['pre_close']
                        # df_i['volume_rate'] = df_i['volume'] / df_i['volume'].shift(1)

                        df_i['open'] = df_i['open'] / close_start
                        df_i['high'] = df_i['high'] / close_start
                        df_i['low'] = df_i['low'] / close_start
                        df_i['close'] = df_i['close'] / close_start
                        df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)

                        # df_i['open'] = np.log(df_i['open'] / df_i['pre_close'])
                        # df_i['high'] = np.log(df_i['high'] / df_i['pre_close'])
                        # df_i['low'] = np.log(df_i['low'] / df_i['pre_close'])
                        # df_i['close'] = np.log(df_i['close'] / df_i['pre_close'])
                        # df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)

                    df_i['return_rate'] = df_i['close'].pct_change().shift(-1)
                    # df_i['return_rate1'] = df_i['close_rate'].pct_change().shift(-1)
                    
                    gpl.set_datas(df_i)
                    index_v = gpl.get_index(index_name, just_index_v=1)
                    index_v.index = df_i.index
                    df_i['index_v'] = index_v 
                    df_i.dropna(inplace=True)
                    # res_li.append(df_i[['index_v', 'return_rate1']].copy())
                    res_li.append(df_i.copy())
                
                df_concat = pd.concat(res_li)
                save_index_pa = makedir(f'{pa_prefix}/simulation/factor_sep_real/{index_name}/')
                df_concat = df_concat.sort_values('index_v', ascending=True)
                # df_concat.reset_index(drop=True, inplace=True)
                df_concat['cumsum_return_rate'] = df_concat['return_rate'].cumsum()
                df_concat['cumprod_return_rate1'] = df_concat['return_rate1'].cumprod()
                df_concat['cumsum_return_rate0'] = df_concat['return_rate0'].cumsum()
                df_concat.to_csv(f'{sy}_concat1.csv')

                if is_plot:
                    for n, plot_name in enumerate(['cumsum_return_rate', 'cumprod_return_rate1', 'cumsum_return_rate0']):
                        plt.figure(figsize=(18, 12))
                        plt.plot(df_concat['index_v'], df_concat[plot_name])
                        # save_index_pa = makedir(f'{pa_prefix}/simulation/factor_sep_real/{index_name}/')
                        plt.savefig(f'{sy}_{plot_name}.png')
                        plt.close()
                        # plt.clf()
                fitness_li.append(df_concat['cumsum_return_rate'].abs().max()-abs(df_concat['cumsum_return_rate'].iloc[-1])/2)
                # print(index_name, sy, fitness_li[-1], 'done.')
            # df_res[index_name] = fitness_li
            print(index_name, num, 'done.')


if __name__ == '__main__':
    # run_simulationoptimizeparams()
    # run_generate_signal()
    # run_optimize_total()
    # run_optimize_total_mltest()
    multi_index_analyze()
    # fitness_plot()
    # diff_df()

    
