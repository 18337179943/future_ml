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
from backtesting.ctabacktester import MyBackTester
from machine_test.ml_test import *
import matplotlib.pyplot as plt
# from gplearn.genetic import SymbolicTransformer
from backtesting.data_analyze_show import plot_pnl_seperate, plot_show1


class SimulationOptimizeParams():

    def __init__(self, symbol='', suffix=None, y_pred=None, strategy_class=BaseStrategy, target_type='total_profit', symbol_li=[]) -> None:
        self.suffix = suffix
        self.save_params = makedir(f'{pa_prefix}/simulation/optuna_params/{symbol}/')
        self.symbol = symbol
        self.y_pred = y_pred
        self.target_type = target_type
        self.symbol_li = symbol_li
        self.sbt = SimulationBackTester(strategy_class=strategy_class)
        self.res_dic = {'symbol': [], 'best_score': [], 'train_score': [], 'val_score': [], 'test_score': [], 'best_params': []}

    def target_func(self, trial):
        '''目标函数'''
        # parameters = {
        #     # 固定参数
        #     'atr_n': trial.suggest_int('atr_n', 6, 24, step=3),
        #     'atr_profit_dev': trial.suggest_float('atr_profit_dev',  0.5, 4, step=0.4),
        #     'atr_loss_dev': trial.suggest_float('atr_loss_dev',  0.2, 
        #                     0.5*trial.suggest_float('atr_profit_dev',  0.5, 4, step=0.4), step=0.5)
        # }

        # trade_mod
        # parameters = {
        #     'stop_loss_n': trial.suggest_int('stop_loss_n', 5, 45, step=3),
        # }

        # trade_mod1
        parameters = {
            # 'acc_n': trial.suggest_int('acc_n', 5, 45, step=3),
            # 'atr_mod1_n': trial.suggest_int('atr_mod1_n', 4, 24, step=3),
            # 'atr_mod1_dev': trial.suggest_float('atr_mod1_dev',  0.3, 2, step=0.2),
            'trade_type': trial.suggest_int('trade_type', 10, 10, step=1),
            # 'stop_loss_n': trial.suggest_int('stop_loss_n', 1, 20, step=1),
            'loss_n': trial.suggest_int('loss_n', 3, 8, step=1)
        }

        # parameters = {
        #     # 固定参数
        #     'atr_n': trial.suggest_int('atr_n', 23,25, step=2),
        #     'atr_profit_dev': trial.suggest_float('atr_profit_dev',  100, 102, step=0.2),
        #     'atr_loss_dev': trial.suggest_float('atr_loss_dev',  100, 102, step=0.2)
        # }
        train_score, val_score, test_score = self.get_train_val_test_score(params=parameters)
        
        target = val_score
        # target = min(train_score, val_score)
        print(f"品种：{self.symbol}, 分数: {target}, 训练集: {train_score}, 验证集: {val_score} -- 测试集: {test_score}")
        # print()

        return target # - abs(target-test_target)

    def get_train_val_test_score(self, params):
        '''获取训练集验证集测试集分数'''
        if self.symbol != 'total':
            _, train_score = self.sbt.all_contract_backtesting(self.symbol, startdate=datetime(2016, 5, 1), enddate=datetime(2019, 5, 1), y_pred=self.y_pred, params=params, target_type=self.target_type)
            _, val_score = self.sbt.all_contract_backtesting(self.symbol, startdate=datetime(2019, 5, 1), enddate=datetime(2020, 5, 1), y_pred=self.y_pred, params=params, target_type=self.target_type)
            _, test_score = self.sbt.all_contract_backtesting(self.symbol, startdate=datetime(2020, 5, 1), enddate=datetime(2020, 10, 31), y_pred=self.y_pred, params=params, target_type=self.target_type)
        else:
            # train_score, _ = self.sbt.all_symbols_backtesting(self.symbol_li, startdate=datetime(2016, 5, 1), enddate=datetime(2019, 5, 1), y_pred_li=self.y_pred, params=params, target_type=self.target_type)
            train_score = 0
            val_score, _ = self.sbt.all_symbols_backtesting(self.symbol_li, startdate=datetime(2019, 5, 1), enddate=datetime(2020, 5, 1), y_pred_li=self.y_pred, params=params, target_type=self.target_type)
            test_score, _ = self.sbt.all_symbols_backtesting(self.symbol_li, startdate=datetime(2020, 5, 1), enddate=datetime(2020, 10, 31), y_pred_li=self.y_pred, params=params, target_type=self.target_type)
        return train_score, val_score, test_score

    def optuna_optimize(self, n_trials):
        '''optuna调参'''
        # try:
        target_direction='maximize'
        study = optuna.create_study(direction=target_direction)
        study.optimize(
            lambda trial : self.target_func(trial), n_trials=n_trials)

        bp = study.best_params
        best_score = study.best_trial.value

        # print('Number of finished trials:', len(study.trials))
        # print("------------------------------------------------")
        # pa_name = 'all' if self.symbol == None else self.symbol
        save_json(bp, f'{self.save_params}best_bp_{self.suffix}.json')
        
        print('Best trial: score {},\nparams {}'.format(best_score,bp))
        print("------------------------------------------------")
        train_score, val_score, test_score = self.get_train_val_test_score(params=bp)
        self.res_dic['symbol'].append(self.symbol)
        self.res_dic['best_score'].append(best_score)
        self.res_dic['best_params'].append(bp)
        self.res_dic['train_score'].append(train_score)
        self.res_dic['val_score'].append(val_score)
        self.res_dic['test_score'].append(test_score)
        del study
        pd.DataFrame(self.res_dic).to_csv(f'{self.save_params}best_bp_{self.suffix}.csv', index=False)
        return self.res_dic

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

def run_optimize_total():
    '''全品种优化'''
    symbol_li, y_pred_li, res_pa, suffix_li, ms = init_val()
    symbol = 'total'
    n_trials = 1
    suffix = 'total'
    print(symbol_li)
    # ['AP', 'FG', 'HC', 'L', 'M', 'pp', 'RM', 'RU', 'AL', 'SF', 'JD', 'JM', 'OI', 'P', 'RB', 'V', 'sn']
    sop = SimulationOptimizeParams(symbol, suffix, y_pred_li, strategy_class=BaseStrategy, target_type='total_profit', symbol_li=symbol_li)   # 'total_profit', 'drawdown', 'max_ddpercent'
    # res_dic = sop.optuna_optimize(n_trials)
    # params = res_dic['best_params'][-1]
    params, suffix = {"trade_type": 9, "stop_loss_n": 2}, 'total'
    
    sop.sbt.signal_analyze_total(symbol_li, y_pred_li, params=params)
    # caculate_statistics(res_pa, symbol, ms, need_signal=0, suffix=suffix)

def run_optimize_total_mltest():
    '''全品种优化'''
    # symbol_li, y_pred_li, res_pa, suffix_li, ms = init_val()
    symbol = 'total'
    n_trials = 17
    suffix = 'total'
    # print(symbol_li)
    # ['AP', 'FG', 'HC', 'L', 'M', 'pp', 'RM', 'RU', 'AL', 'SF', 'JD', 'JM', 'OI', 'P', 'RB', 'V', 'sn']
    mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/symbol_result_10_index_adj/params/')
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
    # params = {"trade_type": 101, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    # params = {"trade_type": 101, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10}
    # params = {"trade_type": 100, "loss_n": 2, 'is_win': 0, 'is_leverage': 0, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10} # 6 10
    # params = {"trade_type": 101, "loss_n": 2, 'is_win': 0, 'is_leverage': 0, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10} # 6 10

    params = {"trade_type": 0}
    # params = {"trade_type": 21}
    # params = {"trade_type": 22}
    # params = {"trade_type": 23, "atr_n_mod23": 4, 'atr_dev_mod23': 7, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 4, 'atr_dev_mod23': 12, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 6, 'atr_dev_mod23': 12, 'is_leverage': 0}
    # params = {"trade_type": 23, "atr_n_mod23": 5, 'atr_dev_mod23': 7, 'is_leverage': 0}
    # params = {"trade_type": 21, "stop_loss_rate": 5/100, 'is_leverage': 0}
    # params = {"trade_type": 22, "stop_loss_rate": 5/100, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 12, 'atr_dev_mod24': 7, 'atr_rate_mod24': 1, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 8, 'atr_dev_mod24': 7, 'atr_rate_mod24': 1, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 7, 'atr_rate_mod24': 0.9, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 5, 'atr_dev_mod24': 3, 'atr_rate_mod24': 0.9, 'is_leverage': 0}
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 2, 'atr_rate_mod24': 0.3, 'is_leverage': 0}  # main
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 5, 'atr_rate_mod24': 0.9, 'atr_dynamic': 1, 'is_leverage': 0}  
    # params = {"trade_type": 24, "atr_n_mod24": 9, 'atr_dev_mod24': 5, 'atr_rate_mod24': 1, 'atr_dynamic': 1, 'is_leverage': 0}  
    # params = {"trade_type": 24, "atr_n_mod24": 5, 'atr_dev_mod24': 5, 'atr_rate_mod24': 0.6, 'atr_dynamic': 1, 'is_leverage': 0}  
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 6, 'atr_rate_mod24': 1, 'atr_dynamic': 1, 'is_leverage': 0}  
    # params = {"trade_type": 24, "atr_n_mod24": 4, 'atr_dev_mod24': 2, 'atr_rate_mod24': 0.3, 'profit_rate_loss': 2, 'atr_dynamic': 0, 'is_leverage': 0}  

    # params = {"trade_type": 24, "atr_n_mod24": 7, 'atr_dev_mod24': 2, 'atr_rate_mod24': 0.3, 'is_leverage': 0}

    # params = {"trade_type": 23, "atr_n_mod23": 4, 'atr_dev_mod23': 1.3, 'is_leverage': 0}

    # params = {"trade_type": 14, "pos_mod": 1, 'atr_n_mod14': 4, 'atr_dev_mod14': 0.1}

    print(params)

    # sop = SimulationOptimizeParams(symbol, suffix, y_pred_li, strategy_class=BaseStrategy, target_type='drawdown', symbol_li=symbol_li)   # 'total_profit', 'drawdown', 'max_ddpercent'
    # res_dic = sop.optuna_optimize(n_trials)
    # params = res_dic['best_params'][-1]
    # # params, suffix = {"trade_type": 9, "stop_loss_n": 2}, 'total'
    # df_res_all = sop.sbt.signal_analyze_total(symbol_li, y_pred_li, params=params)
    # ms = ModelStatistics()

    # ms.caculate_statistics_total(df_res_all, res_save_pa)
    
    # caculate_statistics(res_pa, symbol, ms, need_signal=0, suffix=suffix)
    
    # for i in range(3, 5):
    #     params = {"trade_type": 9, "loss_n": i, 'is_win': 0, 'is_leverage': 0}
    #     val_score, _ = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2019, 5, 1), enddate=datetime(2020, 5, 1), y_pred_li=y_pred_li, params=params, target_type=target_type)
    #     test_score, _ = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2020, 5, 1), enddate=datetime(2020, 10, 31), y_pred_li=y_pred_li, params=params, target_type=target_type)
    #     print(val_score, test_score, i, '-----------')
    # _, df_res_all = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2020, 12, 1), enddate=datetime(2021, 12, 30), 
    #                                         y_pred_li=y_pred_li, params=params, delay=0, target_type='drawdown')
    # df_res_all.reset_index(drop=True, inplace=True)
    # df_res_all.set_index('datetime', inplace=True)
    # df_res_all.plot()
    # plt.show()
    # for i in range(df_res_all.shape[1]):
    #     print()
    # df_res_all.to_csv('df_res_all.csv')
    # print('done')
    # df_res_all['pnl_cost_total'].plot()
    # plt.show()
    df_res_all = sbt.signal_analyze_total(symbol_li, y_pred_li, params=params)
    ms = ModelStatistics()
    ms.caculate_statistics_total(df_res_all, res_save_pa)

    '''检验交易结果是否正确'''
    # df_res, _ = sbt.all_contract_backtesting('AP', startdate=datetime(2020, 5, 1), enddate=datetime(2020, 11, 1), y_pred=y_pred_li[0], 
    #                             params=params, delay=0, target_type=target_type)
    # df_res.to_csv('df_res.csv')
    
def run_generate_signal():
    '''将信号分类'''
    symbol_li, y_pred_li, res_pa, suffix_li, ms = init_val()
    n_trials = 15
    print(symbol_li)
    # ['sn', 'HC', 'RB', 'P', 'JM', 'AP', 'OI', 'JD', 'CF', 'v', 'pp', 'L'] linux
    # ['L', 'pp', 'v', 'CF', 'JD', 'OI', 'AP', 'JM', 'P', 'RB', 'HC', 'sn'] windows
    # ['AP', 'FG', 'HC', 'L', 'M', 'pp', 'RM', 'RU', 'AL', 'SF', 'JD', 'JM', 'OI', 'P', 'RB', 'V', 'sn']
    # i = -2
    # for i in range(9, len(symbol_li)):
    # sbt = SimulationBackTester(strategy_class=SimulationStrategy) # BaseStrategy SimulationStrategy
    # for i in range(len(symbol_li)):
    #     # i = symbol_li.index('AP')
    #     sbt.signal_analyze(symbol_li[i], y_pred_li[i], params={"trade_type": 4, "stop_loss_n": 4})
    #     caculate_statistics(res_pa, symbol_li[i], ms, need_signal=0, suffix='')
    run_type = 'optimize'   # optimize, backtest
    for i in range(len(symbol_li)):  # len(symbol_li)
        # i = symbol_li.index('sn')
        symbol = symbol_li[i]
        # if symbol == 'FG':
        #     continue
        print(symbol)
        sop = SimulationOptimizeParams(symbol, suffix_li[i], y_pred_li[i], strategy_class=SimulationStrategy, target_type='max_ddpercent')   # 'total_profit', 'drawdown'
        # for j in [0.001, 0.003, 0.005]:
        #     for k in [0.01, 0.03, 0.05]:
        j = 0.003
        k = 0.03
        if run_type == 'optimize':
            # 跑优化
            res_dic = sop.optuna_optimize(n_trials)
            params = res_dic['best_params'][-1]
            suffix = ''
        elif run_type == 'backtest':
            # 跑回测
            # ms.caculate_statistics_all(train_pa=train_pa, save_pa=save_pa, symbol=symbol)
            df_statis = caculate_statistics(res_pa, symbol, ms)
            hold_n = int(np.mean(df_statis[df_statis.index == '平均持仓周期'].iloc[:, :2].values))
            params = {'trend_n': hold_n*4, 'revers_n': hold_n, 'signal_thread1': j, 'signal_thread2': k} # 
            suffix = f'{j}_{k}'
        else:
            params, suffix = {"trade_type": 8, "stop_loss_n": 4}, ''
            
        sop.sbt.signal_analyze(symbol, y_pred_li[i], params=params)
        caculate_statistics(res_pa, symbol, ms, need_signal=0, suffix=suffix)

def run_simulation_train_val_test(pa=''):
    '''全品种优化'''
    symbol = 'total'
    ind = 0
    is_params_save = 0
    folder_name = 'adj_target_sort_10_16/model_raw'
    if len(pa) == 0:
        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std/model_raw/model_2.0_6same/'
        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_sum_values_4/model_raw/model_2.0/'
        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/{folder_name}/model_2.0.{ind}/'
        pa = f'{pa_prefix}/datas/ml_result/model_2.0/{folder_name}/model_2.0/'
        # pa = f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/'

        # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_early_stop/model_zigzag/model_2.0/'
        # pa = f'{pa_prefix}/datas/ml_result/model_1.0/model_zigzag/raw12/raw/'
        pa1 = f'{pa_prefix}/datas/ml_result/model_2.0/{folder_name}/'
        # pa1 = f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/'

    # print(symbol_li)
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/symbol_result_10_index_adj/params/')
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_early_stop/model_raw/model_2.0_raw/')
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_early_stop/model_zigzag/model_2.0_zigzag/')
    print(pa)
    res_save_pa = makedir(f'{pa1}{symbol}_train{ind}/')
    mlt = MLTest(load_pa=f'{pa}', is_params_save=is_params_save)
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/symbol_result_10_index_adj1/params/')
    symbol_li = mlt.symbols_li
    y_pred_li = mlt.generate_y_pred_total()
    # res_save_pa = makedir(f'{pa_prefix}/simulation/optuna_params/{symbol}/')

    sbt = SimulationBackTester(strategy_class=SimulationStrategy)
    target_type = 'max_ddpercent'
    params = {"trade_type": 9, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    params = {"trade_type": 10, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    params = {"trade_type": 13, "atr_n_mod13": 4, 'atr_dev_mod13': 4/10}
    params = {"trade_type": 17, "loss_n": 2, 'is_win': 0, 'is_leverage': 0} # 6 10
    params = {"trade_type": 0}
    print(params)

    df_res_all = sbt.signal_analyze_total(symbol_li, y_pred_li, params=params, sp=res_save_pa)
    ms = ModelStatistics()
    ms.caculate_statistics_total(df_res_all, res_save_pa, prefix='_total')
    print('done.')

def run_simulation_test():
    '''全品种在某段时间上回测'''
    symbol = 'total'
    is_params_save = 1
    # startdate, enddate = datetime(2020, 12, 1), datetime(2022, 12, 14)
    startdate, enddate = datetime(2016, 1, 1), datetime(2020, 12, 1)
    # pa = f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index_adj1/params/raw/'
    pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std_3/model_zigzag/model_2.0/'
    # pa = f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_sum_values_4/model_raw/model_2.0/'
    pa = f'{pa_prefix}/datas/ml_result/model_2.0/original_factor_skew_product_5/raw1/model_2.0/'
    # pa = makedir(f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/')
    # pa = f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/'
    pa = makedir(f'{pa_prefix}/datas/ml_result/model_2.0/adj_target_sort_10_16/model_raw/model_2.0/')

    res_save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_2.0/adj_target_sort_10_16/model_raw/{symbol}_test2016/')
    # res_save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_2.0/skew_skew_10_13/model_raw/{symbol}_train1/')
    # res_save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/{symbol}_test/')
    # res_save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/test2022/')
    # exit()
    print(res_save_pa)
    mlt = MLTest(load_pa=pa, is_params_save=is_params_save, enddate=enddate)
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/model_2.0/')
    # mlt = MLTest(load_pa=f'{pa_prefix}/model_2.0/model_zigzag/model_2.0_zigzag/')
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/model_2.0/factor_sort_std/model_raw/model_2.0/')
    symbol_li = mlt.symbols_li
    y_pred_li = mlt.generate_y_pred_total()

    sbt = SimulationBackTester(strategy_class=SimulationStrategy, enddate=enddate)
    # sbt = MyBackTester()
    params = {"trade_type": 0}

    # for i in range(len(symbol_li)):
    #     y_pred_li[i].to_csv('y_pred.csv')
    #     print(symbol_li[i])
    #     input()

    print(params)

    _, df_res_all = sbt.all_symbols_backtesting(symbol_li, startdate=startdate, enddate=enddate, 
    # _, df_res_all = sbt.all_symbols_backtesting(symbol_li, startdate=datetime(2022, 8, 18), enddate=enddate, 
                                            y_pred_li=y_pred_li, params=params, delay=20, target_type='drawdown', save_pa=res_save_pa)
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
        pa = res_save_pa + filter_str(f'_{sy}.csv', symbol_analyze_li)
        plot_show1(sy, pa=pa, save_pa=f'{res_save_pa}analyze_{sy}.jpg', mod=3)
    ms.get_trade_result(res_save_pa)
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy2}.csv', suffix=f'_{sy2}')
    # ms.caculate_statistics_single(df_res_all.copy(), save_pa=f'{res_save_pa}statistic_{sy3}.csv', suffix=f'_{sy3}')
    print('done.')
    return 

if __name__ == '__main__':
    # run_simulationoptimizeparams()
    # run_generate_signal()
    # run_optimize_total()
    # run_optimize_total_mltest()
    run_simulation_test()
    # run_simulation_train_val_test()
