import imp
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
import numpy as np
import pandas as pd
from search_factor.factor_process import *
from datas_process.m_futures_factors import FactorIndexStatistics
from datas_process.m_datas_process import *
from deep_learning.lstmmodel import *
from machine_learning.optimize_params1 import run_dp_optimize_all   # 数据处理和模型优化

__Author__ = 'ZCXY'

'''
深度学习步骤:
1、进行因子分析，选出因子。
2、对因子根据因子对数据进行处理，得到x和y
3、对x和y进行标准化和标签化
4、划分训练集，验证集和测试集
5、训练模型和保存结果
6、对模型进行回测
'''

def main_model_run():
    # syinfo = SymbolsInfo()
    symbol_li = ['AP', 'AG', 'AL', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 
                'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 
                'SF', 'SN', 'SR', 'TA', 'V', 'Y', 'ZN']
    # threadhold_li = [1.2, 1.3, 1.4, 1.6, 1.8]
    # threadhold_li = [0.7, 0.9, 1.1, 1.3, 1.5]
    threadhold_li = [0.2, 0.4, 0.6, 0.8]
    y_thread_li = [[5, 0.5, 1, 1], [7, 0.6, 1, 1], [10, 0.5, 1, 1], [10, 1, 1, 1]]
    
    for symbol in symbol_li[:3]:
        print('begin: ', symbol)
        run_factorprocess_all(threadhold_li, [symbol], is_symbol_factor=1)  # 因子筛选
        run_dp_optimize_all(symbol, threadhold_li, y_thread_li, n_trials=300)   # 因子数据处理和模型训练
        print('end: ', symbol)
    

if __name__ == '__main__':
    main_model_run()        # 因子筛选和模型训练
    # run_concatstatistics()      # 统计训练结果
    # run_simulation_train_val_test()     # 全品种组合在训练集验证集测试集上回测
    # run_simulation_test()       # 全品种在某段时间上回测

