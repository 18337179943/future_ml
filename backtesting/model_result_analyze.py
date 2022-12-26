import sys
from m_base import *
sys_name = 'windows'
pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from datetime import datetime
from datas_process.m_futures_factors import SymbolsInfo, MainconInfo
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
# from atr_rsi_strategy import AtrRsiStrategy
from m_base import Logger, get_sy, timestamp_to_datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
sys.path.append('..')


class ModelAnalyze:
    '''模型结果分析'''
    def __init__(self, interval, startdate=datetime(2016, 7, 20), enddate=datetime(2020, 10, 30)) -> None:
        self.pred_pa = f'{pa_prefix}/datas/predict/'
        self.datas_pa = f'{pa_prefix}/datas/data_{interval}m'
        self.interval = interval
        self.mainconinfo = MainconInfo()
        self.startdate = startdate
        self.enddate = enddate

    def signal_analyze(self, symbol, pred_pa, datas_pa):
        '''信号分析'''
        pass
