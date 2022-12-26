from mimetypes import suffix_map
from operator import index
from posixpath import split
from symtable import Symbol
from tkinter import SEL, TRUE
from tkinter.tix import Tree
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
from datetime import datetime, timedelta
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
# os.path.append('..')
from datas_process import m_futures_factors as mff 
from m_base import sharp_ratio
from datas_process import m_datas_process as mdp
from datas_process.m_futures_factors import MainconInfo, SymbolsInfo
from m_base import makedir
import pandas_ta as ta
from math import *
from functools import partial
from backtesting.data_analyze_show import plot_show_index

# s = np.array([1,2,2,2])
# d = (s,s,s,s)
# df = pd.DataFrame()
# df['dd'] = [1,1,1,1]
# df[[f'{i}' for i in range(len(d))]] = d
# df

class FactorAnalyze():
    '''因子分析，画k线图和因子'''
    def __init__(self, startdate=datetime(2018, 1, 1), enddate=datetime(2018, 7, 30)) -> None:
        self.pa = f'{pa_prefix}/datas/data_index/'
        self.startdate, self.enddate = startdate, enddate
        self.bar_index = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        self.factorindex = mff.FactorIndex
        self.mainconinfo = mff.MainconInfo()
    
    def get_index_name(self):
        '''获取指标名称'''
        index = dir(self.factorindex)
        return [i for i in index if '__' not in i]

    def get_symbol_factor_data(self, symbol, win_n, is_pa, func_name=None):
        '''获取bar和指标, func_name: list, win_n: list'''
        self.func_name = func_name
        if is_pa:
            sy_pa = f'{self.pa}{symbol}/'   # 从文件里读取指标
            df = pd.read_csv(f'{sy_pa}{symbol}_60m.csv')
            df['datetime'] = pd.to_datetime(df['datetime'])
            if func_name is not None:
                col_li = self.bar_index + func_name
                df = df[col_li]
                df = df[(df['datetime'].dt.date >= self.startdate.date()) & (df['datetime'].dt.date < self.enddate.date())]
        else:
            df_li = self.mainconinfo.get_main_contact_k_line(symbol, self.startdate, self.enddate, delay=20, load_pa=None, is_concat=0)
            df_res_li = []
            if func_name[0] == 'pandas_ta':
                for df_i in df_li:
                    fi = self.factorindex(df_i)
                    for index_i, win_i in zip(func_name[1:], win_n):
                        df_i, column = fi.pandas_ta(df_i, win_i, index_i)
                    df_res_li.append(df_i)
            elif func_name[0] == 'talib':
                for df_i in df_li:
                    fi = self.factorindex(df_i)
                    for index_i, win_i in zip(func_name[1:], win_n):
                        func = getattr(fi, index_i)
                        index_res = func(*win_i)
                        if isinstance(index_res, tuple):
                            for i in range(len(index_res)):
                                df_i[f'{index_i}_{i}'] = index_res[i]
                            # df_i[[f'{fn}_{i}' for i in range(len(index_res))]] = index_res 
                        else:
                            df_i[index_i] = index_res
                    df_res_li.append(df_i)
                
            df_concat = pd.concat(df_res_li)
            df_concat = df_concat.drop_duplicates(subset=['datetime'], keep='first')
            df_concat.fillna(method='ffill', inplace=True)
            df_concat.dropna(inplace=True)
            df = df_concat
                    
        return df

    def plot_price_factor(self, symbol, df, save_pa=None, mod=0):
        '''画图'''
        plot_show_index(symbol, df, save_pa=save_pa, mod=mod)

    def main(self, symbol, win_n, is_pa, func_name=None, save_pa=None, mod=0):
        '''主函数，先计算指标，后画图'''
        df = self.get_symbol_factor_data(symbol, win_n, is_pa, func_name)
        self.plot_price_factor(symbol, df, save_pa, mod)


def run_FactorAnalyze():
    pa = ''
    symbol = 'RB'
    func_name = ['ad_11', 'adosc_11']
    index_type = ['pandas_ta', 'talib']
    func_name = ['talib', 'sma', 'kama']
    # win_n = [12, 26, 9]
    win_n = [[12], [26]]
    is_pa = 0
    save_pa = None
    fa = FactorAnalyze()
    mod = 1
    fa.main(symbol, win_n, is_pa, func_name, save_pa, mod)


if __name__ == "__main__":
    run_FactorAnalyze()


    
