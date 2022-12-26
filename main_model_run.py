import os
from turtle import done
from datas_process.m_datas_process import run_nkl_all
from m_base import filter_str       # 合成k线
from search_factor.factor_process import run_factorprocess_all      # 因子筛选
from machine_learning.optimize_params1 import run_dp_optimize_all   # 数据处理和模型优化
from machine_learning.trainmodel import BaseModel   
from datas_process.m_futures_factors import SymbolsInfo 
from backtesting.model_statistics import run_concatstatistics
from simulation.simulation_optimize_params import run_simulation_train_val_test, run_simulation_test
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from time import sleep



def main_model_run(sy_li):
    # syinfo = SymbolsInfo()
    symbol_li = ['AP', 'AG', 'AL', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 
                'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 
                'SF', 'SN', 'SR', 'TA', 'V', 'Y', 'ZN']
    # threadhold_li = [1.2, 1.3, 1.4, 1.6, 1.8]
    threadhold_li = [0.7, 0.9, 1.1, 1.3, 1.5]
    # threadhold_li = [0.84, 1.08, 1.32, 1.56, 1.8]
    threadhold_li = [1.3, 1.5]
    # threadhold_li = [0.5, 0.6, 0.8, 1.0, 1.2]
    # threadhold_li = [0.2, 0.4, 0.6, 0.8]
    # threadhold_li = [0.7*1.5, 0.9*1.5, 1.1*1.5, 1.3*1.5, 1.5*1.5]

    y_thread_li = [[5, 0.5, 1, 1], [7, 0.6, 1, 1], [10, 0.5, 1, 1], [10, 1, 1, 1]]
    y_thread_li = [[10, 1, 1, 1]]
    for symbol in [sy_li]:
        print('begin: ', symbol)
        run_factorprocess_all(threadhold_li, [symbol], is_symbol_factor=0)  # 因子筛选
        run_dp_optimize_all(symbol, threadhold_li, y_thread_li, n_trials=300, is_optuna=1)   # 因子数据处理和模型训练
        print('end: ', symbol)

def child_optimize(optimize_func=main_model_run):
    '''跑优化的子进程'''
    max_workers = 1
    # symbol_li = ['AP', 'AG', 'AL']
    symbol_li = ['AP', 'PP', 'M']
    left_symbol_li = list(filter(lambda x: x not in os.listdir(f'./datas/ml_result/'), symbol_li))
    optimize_func(left_symbol_li[0])
    # if len(left_symbol_li):
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:  # max_workers=10
    #         results = executor.map(optimize_func, [left_symbol_li[0]])
    
    print('子进程关闭成功1')

def parant_optimize(child_func=child_optimize):
    '''跑优化的父进程'''
    # symbol_li = ['AP', 'AG', 'AL']
    symbol_li = ['AP', 'PP', 'M']
    while len(symbol_li):
        child_process = multiprocessing.Process(target=child_func)
        child_process.start()
        while True:
            if not child_process.is_alive():
                child_process = None
                print('子进程关闭成功2')
                break
            else:
                sleep(2)
        symbol_li = list(filter(lambda x: x not in os.listdir(f'./datas/ml_result/'), symbol_li))
        
    

if __name__ == '__main__':
    # BaseModel().del_model_file()
    # parant_optimize()
    main_model_run('RB')        # 因子筛选和模型训练
    # run_concatstatistics()      # 统计训练结果
    # run_simulation_train_val_test()     # 全品种组合在训练集验证集测试集上回测
    # run_simulation_test()       # 全品种在某段时间上回测
