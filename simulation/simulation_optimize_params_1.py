from cgi import test
from cmath import nan
import imp
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
from simulation.base_strategy import BaseStrategy
from backtesting.ml_strategy_adj import MLStrategy
import shutil
from backtesting.model_statistics import ModelStatistics
from machine_test.ml_test import *
import matplotlib.pyplot as plt
from backtesting.data_analyze_show import plot_pnl_seperate, plot_show1


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
    # print(symbol_li)
    # ['AP', 'FG', 'HC', 'L', 'M', 'pp', 'RM', 'RU', 'AL', 'SF', 'JD', 'JM', 'OI', 'P', 'RB', 'V', 'sn']
    mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/symbol_result_10_index_adj1/params/')
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/total/')
    # mlt = MLTest()
    symbol_li = mlt.symbols_li
    y_pred_li = mlt.generate_y_pred_total()
    res_save_pa = makedir(f'{pa_prefix}/simulation/optuna_params/{symbol}/')

    sbt = SimulationBackTester(strategy_class=SimulationStrategy)
    target_type = 'max_ddpercent'
    params = {"trade_type": 9, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    params = {"trade_type": 10, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 11, "stop_loss_n": 2, 'is_leverage': 0}
    # params = {"trade_type": 10, "loss_n": 6, 'is_win': 1, 'is_leverage': 0}
    # params = {"trade_type": 10, "loss_n": 2, 'is_win': 1, 'is_leverage': 0}
    # params = {"trade_type": 13, "loss_n": 3, 'is_win': 0, 'is_leverage': 0}
    # params = {"trade_type": 10, "loss_n": 1000, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 12}
    params = {"trade_type": 13, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10}
    params = {"trade_type": 17, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 15, "atr_n_mod13": 4, 'atr_dev_mod13': 5/10}
    # params = {"trade_type": 16, 'limit_n': 1}
    # params = {"trade_type": 100, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 0, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 101, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 101, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10}
    # params = {"trade_type": 100, "loss_n": 2, 'is_win': 0, 'is_leverage': 0, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10} # 6 10
    # params = {"trade_type": 101, "loss_n": 2, 'is_win': 0, 'is_leverage': 0, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10} # 6 10
    # params = {"trade_type": 19, "atr_n_mod19": 8, 'atr_dev_mod19': 11/10, 'is_leverage': 0}
    params = {"trade_type": 20, 'is_win': 0, 'is_leverage': 0} # 6 10

    params = {"trade_type": 0}
    # params = {"trade_type": 21}
    # # params = {"trade_type": 22}
    # params = {"trade_type": 23, "atr_n_mod23": 12, 'atr_dev_mod23': 1.3, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 4, 'atr_dev_mod23': 12, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 6, 'atr_dev_mod23': 12, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 5, 'atr_dev_mod23': 7, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 12, 'atr_dev_mod23': 7, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 12, 'atr_dev_mod24': 7, 'atr_rate_mod24': 1, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 15, 'atr_rate_mod24': 0.9, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 2, 'atr_rate_mod24': 0.3, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 2, 'atr_rate_mod24': 0.3, 'profit_rate_loss': 2, 'atr_dynamic': 0, 'is_leverage': 0}  
    # params = {"trade_type": 24, "atr_n_mod24": 5, 'atr_dev_mod24': 5, 'atr_rate_mod24': 0.6, 'profit_rate_loss': 500, 'atr_dynamic': 1, 'is_leverage': 0}  # test
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 2, 'atr_rate_mod24': 0.3, 'profit_rate_loss': 500, 'atr_dynamic': 0, 'is_leverage': 0}  # test

    # params = {"trade_type": 14, "pos_mod": 1, 'atr_n_mod14': 4, 'atr_dev_mod14': 0.1}

    print(params)

    _, df_res_all = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2022, 1, 1), enddate=datetime(2022, 8, 16), 
                                            y_pred_li=y_pred_li, params=params, delay=20, target_type='drawdown', save_pa=res_save_pa)
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
        plot_show1(sy, pa=pa, save_pa=f'{res_save_pa}analyze_{sy}.jpg', mod=3)
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy2}.csv', suffix=f'_{sy2}')
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy3}.csv', suffix=f'_{sy3}')
    print('done.')

    '''检验交易结果是否正确'''
    # df_res, _ = sbt.all_contract_backtesting('AP', startdate=datetime(2020, 5, 1), enddate=datetime(2020, 11, 1), y_pred=y_pred_li[0], 
    #                             params=params, delay=0, target_type=target_type)
    # df_res.to_csv('df_res.csv')

if __name__ == '__main__':
    # run_simulationoptimizeparams()
    # run_generate_signal()
    # run_optimize_total()
    run_optimize_total_mltest()
