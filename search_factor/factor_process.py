import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import partial
import sys, os

from pyautogui import PRIMARY
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
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from math import *
__Author__ = 'ZCXY'
# from mmm import msm


class FactorProcess(mdp.BaseDataProcess):
    '''因子处理'''
    def __init__(self, startdate=datetime(2016, 1, 1), enddate=datetime(2022, 12, 14), interval=[60], traindate=datetime(2020, 5, 1), fi_method=1):  # datetime(2020, 11, 10)
    # def __init__(self, startdate=datetime(2016, 1, 1), enddate=datetime(2022, 8, 1), interval=[60], traindate=datetime(2020, 5, 1)):  # datetime(2020, 11, 10)
        self.interval_li = interval       # k线周期
        self.win_n = [i for i in range(3, 16, 4)] if interval[-1] != 60 else [i for i in range(6, 25, 5)] # 指标周期 h  # [i for i in range(6, 25, 5)]
        self.pred_h = [60]    # y的预测周期 [60] 
        self.fi_class = mff.FactorIndex if fi_method == 0 else mff.MyFactorIndex
        self.index_class = [3, 4, 5]   # y的分类
        self.y_list = self.y_name()     # y的种类名称（4种）
        self.startdate=startdate        # 开始时间
        self.enddate=enddate            # 结束时间
        self.save_index_pa = f'{pa_prefix}/datas/factors_analyze/index_name/'
        self.save_res_pa = f'{pa_prefix}/datas/factors_analyze/'
        self.maincon = self.get_maincon()   # 每日品种主力合约列表
        self.index_name = self.get_index_name()  # 指标名称 
        self.index_name_n = []  # 指标名称 
        res_dict = self.init_dict()        # 初始化统计结果的字典
        self.columns_original = []
        self.traindate = traindate

        self.maincon_info = MainconInfo()
        self.syinfo = SymbolsInfo()
        self.symbol_li = self.syinfo.symbol_li
        self.no_stationary_li = self.get_no_stationary_li()
        # self.index_li = self.get_index_li()
        self.columns_original = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'return_rate_60m', 'return_rate']
        self.hist_res_method = 2   # 计算收益分布的方法
        self.need_del_stationary = 1

    def get_no_stationary_li(self, pa=f'{pa_prefix}/datas/factors_analyze/all/del/'):
        ''''''
        pa_li = os.listdir(pa)
        no_stationary_li = []
        for pa_i in pa_li:
            if '.csv' in pa_i: continue
            pa_i = pa_i[:-4]
            index_i = ''
            for j in pa_i.split('_'):
                if len(check_letters(j)): index_i += f'{j}_'
            no_stationary_li.append(index_i)
        no_stationary_li = no_stationary_li + ['atr_', 'natr_', 'std_', 'sub_', 'var_']
        # pd.DataFrame(no_stationary_li).to_csv('df_no_stationary_li.csv')
        return no_stationary_li

    def get_index_li(self):
        index_li = None
        if not os.path.exists(self.save_index_pa):
            makedir(self.save_index_pa)
        else:
            if len(os.listdir(self.save_index_pa)) != 0:
                df = pd.read_csv(self.save_index_pa+'index.csv')
                index_li = df['index'].to_list()
        return index_li

    def init_dict(self):
        '''初始化统计结果的字典'''
        res_dict = {'columns': [], 'index': [], 'index_n': [], 'interval': [],  'y_name': [], 'class': [], 'corr': [], 'smooth': []}
        for i in range(np.max(self.index_class)):
            res_dict.update({f'deviation{i}': [], f'skew{i}': [], f'kurt{i}': []})
        return res_dict

    def del_no_stationary_index(self, adf_res):
        ''''''
        for i in self.no_stationary_li:
            index_li = filter_str(i, adf_res, is_list=1)
            [adf_res.remove(index_i) for index_i in index_li]
        return adf_res

    def get_maincon(self):
        '''获取每日主力合约'''
        df = pd.read_csv(f'{pa_prefix}/datas/maincon.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= self.startdate) & (df['date'] <= self.enddate)]
        return df

    def get_index_name(self):
        '''获取指标名称'''
        index = dir(self.fi_class)
        return [i for i in index if '__' not in i]
    
    def y_name(self):
        '''y名称'''
        y_list = []
        for i in self.pred_h:
            y_list.append([f'return_rate_{i}m'])
            # y_list.append([f'return_rate_{i}m', f'mean_return_rate_{i}m'])
        return sum(y_list, [])

    def get_index(self, df: pd.DataFrame, index_func, name, n):
        '''获取指标'''
        func = getattr(index_func, name)
        params_n = func.__code__.co_argcount    # 判断指标需要输入的参数
        need_break = 0
        if params_n == 1:
            res = func()
        else:
            try:
                res = func(n)
            except:
                return df, need_break
        index_name = f'{name}_{n}'
        # if type(res) is not tuple:
        #     df[index_name] = res
        # else:
        #     for i, j in enumerate(res):
        #         df[f'{index_name}_{i}'] = j
        df[index_name] = res if type(res) is not tuple else res[0]    # 判断返回的参数

        if len(df[index_name].unique()) <= 5:  # 指标值小于5的直接去掉
            del df[index_name]
            need_break = 1
        # elif index_name not in self.index_name_n:
        #     self.index_name_n.append(index_name)
        
        return df, need_break

    def _caculate_index(self, df, win_n):
        '''计算所有指标'''
        index_func = self.fi_class(df)
        for name in self.index_name:    # talib每个指标名称
            for n in win_n:   # 时间窗口
                df, need_break = self.get_index(df, index_func, name, n)
                if need_break:
                    break
        
        for name in index_func.pdtb:
            for n in win_n:     # pandas.ta指标
                df, index_name_li = index_func.pandas_ta(df, n, name)
                if index_name_li == '':
                    break

                for index_name in index_name_li:
                    if len(df[index_name].unique()) <= 20:  # 指标值小于5的直接去掉
                        del df[index_name]
                    # elif index_name not in self.index_name_n:
                    #     self.index_name_n.append(index_name)
        return df

    def interval_factor(self, symbol, interval):
        '''获取一个品种一个周期的所有指标'''
        self.index_name_n = []
        df_maincon = self.maincon[self.maincon['symbol']==symbol.upper()]
        # contracts = set(df_maincon['contract'].to_list())
        load_pa = f'{pa_prefix}/datas/data_{interval}m/{symbol}/'
        df_contract_info = self.maincon_info.get_symbol_df_maincon(symbol, self.startdate, self.enddate)
        contracts = df_contract_info['contract'].to_list()
        win_n = list(np.array(self.win_n)*60//interval) if interval != 60 else self.win_n
        df_all = []
        for contract in contracts:      # 品种的合约
            df = pd.read_csv(f'{load_pa}{contract}.csv')
            df_contract = df_maincon[df_maincon['contract']==contract]
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(contract)
            if df['datetime'].iloc[0] > df_contract['date'].iloc[0] - timedelta(days=15):
                print('del ', contract)
                continue
            df = df[(df['datetime'].dt.date >= df_contract['date'].iloc[0] - timedelta(days=15)) & (df['datetime'].dt.date <= df_contract['date'].iloc[-1])]
            df = self.get_pred_y(df, interval)
            # self.columns_original = df.columns.to_list()
            df = self._caculate_index(df, win_n)    # 计算指标
            df = df[(df['datetime'] >= df_contract['date'].iloc[0]) & (df['datetime'] <= df_contract['date'].iloc[-1])]
            df_all.append(df)
        df_all = pd.concat(df_all)
        return df_all

    def get_pred_y(self, df: pd.DataFrame, interval):
        '''获取y值'''
        df['return_rate'] = df['close'].pct_change()
        for i in self.pred_h:
            n = i // interval
            df[f'return_rate_{i}m'] = df['close'].pct_change(n).shift(-n)
            # df[f'sharp_ratio_{i}m'] = df[f'return_rate_{i}m'].rolling(n).apply(sharp_ratio).shift(-n)
            # df[f'mean_return_rate_{i}m'] = df['close'].pct_change().rolling(n).mean().shift(-n)

        df.dropna(inplace=True)
        return df

    def process_df_before_analyze(self, df0: pd.DataFrame):
        '''指标分析前进行的数据处理'''
        df = df0.copy()
        df.fillna(method='ffill', inplace=True)
        df.dropna(axis=1, inplace=True)
        index_li = df.columns.to_list()
        [index_li.remove(i) for i in self.columns_original]

        # adf_res = index_li
        # print('index_li', len(index_li))
        # input()
        df_index = df[index_li].copy()
        return df_index, df, index_li

    def factors_analyze(self, df: pd.DataFrame, interval, symbol, res_dict):
        '''指标分析'''
        df_index, df, index_li = self.process_df_before_analyze(df)
        df_params = self.caculate_skew_mean_std(df, symbol, interval)   # 计算偏度相关参数
        # print(df_index.shape)
        
        adf_res = self.select_adf_test_res(df_index)    # 选出平稳序列的指标

        if self.need_del_stationary: adf_res = self.del_no_stationary_index(adf_res)

        # print('adf_res', len(adf_res))
        # input()
        # print('adf_res', adf_res)
        # print(len(adf_res))
        # df[adf_res].to_csv('df0.csv')

        for index in index_li:
            # print('start:', index)
            # st = datetime.now()
            if self.hist_res_method == 0:
                res_dict = self.hist_res(df, index, interval, adf_res, res_dict)
            elif self.hist_res_method == 1:     # 用所有品种计算出来的范围做分布
                res_dict = self.hist_res1(df, index, interval, adf_res, res_dict)
            elif self.hist_res_method == 2:     # 用单个品种计算出来的范围做分布
                res_dict = self.hist_res2(df, index, interval, adf_res, res_dict, symbol)
            # print('end:', datetime.now()-st)
        
        columns = self.columns_original + adf_res

        return df, res_dict, df_params

    def hist_res(self, df0: pd.DataFrame, index, interval, adf_res, res_dict):
        '''计算分布情况'''
        # df0.to_csv('df0.csv')
        # exit()
        df_train = df0[df0['datetime'].dt.date < self.traindate.date()]     # 还没有测试
        index_n = index.split('_')[-1]
        # real_index = index[:-len(index_n)-1]
        real_index = index.replace(f'_{index_n}', '')
        q = df_train[index].copy()
        for cla in self.index_class:    # y的分类
            step = 100 // cla
            # res_dict['index'].append(index), res_dict['interval'].append(interval)
            df = df_train.copy()
            df['quantile'] = np.nan
            overlap = 0

            for i in range(0, cla*step, step):  # 对指标进行分类
                if q.quantile(i/100) == q.quantile((i+step)/100):
                    overlap = 1
                df['quantile'] = np.where((q>=q.quantile(i/100)) & (q<=q.quantile((i+step)/100)), i, df['quantile'])
            
            if overlap or len(df['quantile'].value_counts()) != cla:
                print('overlap:', index)
                continue

            df.dropna(inplace=True)
            # print('quantile', df['quantile'].unique())
            
            for y_i in self.y_list:
                adf_n = 1 if index in adf_res else 0
                res_dict['smooth'].append(adf_n)
                res_dict['columns'].append(index), res_dict['corr'].append(np.sign(df[y_i].corr(q)))
                res_dict['index'].append(real_index), res_dict['index_n'].append(index_n)
                res_dict['interval'].append(interval), res_dict['class'].append(cla)
                res_dict['y_name'].append(y_i)

                for i, data in enumerate(df.groupby('quantile')):  # 计算每个分类对应的y的分布
                    data = list(data)[1]    
                    len_data = len(data)
                    # print(data)
                    # print(len(data[data[y_i]>0]) / len_data)
                    # print(data[y_i].skew())
                    # print(data[y_i].kurt())
                    res_dict[f'deviation{i}'].append(len(data[data[y_i]>0]) / len_data)
                    res_dict[f'skew{i}'].append(data[y_i].skew())
                    res_dict[f'kurt{i}'].append(data[y_i].kurt())
                
                # if i != (cla-1):
                #     df[[index, 'quantile']].to_csv('q.csv', index=False)

                if cla != np.max(self.index_class):  # 把少分类的补齐
                    for i in range(cla, np.max(self.index_class)):
                        res_dict[f'deviation{i}'].append(0)
                        res_dict[f'skew{i}'].append(0)
                        res_dict[f'kurt{i}'].append(0)

                if len(res_dict[f'deviation{0}']) != len(res_dict[f'deviation{4}']):
                    print(i, y_i)
                    print('index', len(res_dict['index']))
                    print('index_n', len(res_dict['index_n']))
                    print('interval', len(res_dict['interval']))
                    print('class', len(res_dict['class']))
                    print('y_name', len(res_dict['y_name']))
                    print('deviation0', len(res_dict['deviation0']))
                    print('deviation1', len(res_dict['deviation1']))
                    print('deviation2', len(res_dict['deviation2']))
                    print('deviation3', len(res_dict['deviation3']))
                    print('deviation4', len(res_dict['deviation4']))
                    print('skew0', len(res_dict['skew0']))
                    print('skew1', len(res_dict['skew1']))
                    print('skew2', len(res_dict['skew2']))
                    print('skew3', len(res_dict['skew3']))
                    print('skew4', len(res_dict['skew4']))
                    print('kurt0', len(res_dict['kurt0']))
                    print('kurt1', len(res_dict['kurt1']))
                    print('kurt2', len(res_dict['kurt2']))
                    print('kurt3', len(res_dict['kurt3']))
                    print('kurt4', len(res_dict['kurt4']))
                    print(res_dict['deviation4'])
                    print('------------')
                    # print(res_dict['deviation3'])
                    input()
        return res_dict
    
    def load_df_params(self, symbol='all'):
        '''获取因子参数，最大最小值，中值'''
        if symbol != 'all':
            df_params_all = pd.read_csv(f'{pa_prefix}/datas/factors_analyze/all/all_60m_params_merge_adj.csv')
            df_params_sy = pd.read_csv(f'{pa_prefix}/datas/data_index/{symbol}/{symbol}_60m_params_merge.csv')
        else:
            df_params_all = pd.read_csv(f'{pa_prefix}/datas/factors_analyze/all/all_60m_params_merge_adj.csv')
            df_params_sy = df_params_all

        return df_params_sy, df_params_all 

    def hist_res1(self, df0: pd.DataFrame, index, interval, adf_res, res_dict):
        '''计算分布情况'''
        df_params, _ = self.load_df_params()
        df_train = df0[df0['datetime'].dt.date < self.traindate.date()]     # 还没有测试
        index_n = index.split('_')[-1]
        # real_index = index[:-len(index_n)-1]
        real_index = index.replace(f'_{index_n}', '')
        q = df_train[index].copy()
        df_index = df_params[df_params['index_name']==index]
        if len(df_index['range_max']):
            range_max, range_min = df_index['range_max'].iloc[0], df_index['range_min'].iloc[0]
        else:
            print(index, 'not in range')
            return res_dict

        for cla in self.index_class:    # y的分类
            step = (range_max-range_min) / cla
            df = df_train.copy()
            df['quantile'] = np.nan

            for i in range(0, cla):  # 对指标进行分类
                q_min, q_max = range_min+i*step, range_min+(i+1)*step
                df['quantile'] = np.where((q>=q_min) & (q<=q_max), i, df['quantile'])

                # if index == 'atrzc_11':
                #     print(index, q_max, q_min, range_min, range_max, q.max(), q.min(), len(df['quantile'].value_counts()))
                #     input()
            
            real_cla_li = df['quantile'].value_counts().index.to_list()
            
            if len(real_cla_li) == 0:
                print('overlap:', index)
                continue

            df.dropna(inplace=True)
            # print('quantile', df['quantile'].unique())
            
            for y_i in self.y_list:
                adf_n = 1 if index in adf_res else 0
                res_dict['smooth'].append(adf_n)
                res_dict['columns'].append(index), res_dict['corr'].append(np.sign(df[y_i].corr(q)))
                res_dict['index'].append(real_index), res_dict['index_n'].append(index_n)
                res_dict['interval'].append(interval), res_dict['class'].append(cla)
                res_dict['y_name'].append(y_i)

                for data in df.groupby('quantile'):  # 计算每个分类对应的y的分布
                    cla_i = int(list(data)[0])    
                    data = list(data)[1]    
                    len_data = len(data)
                    # print(data)
                    # print(len(data[data[y_i]>0]) / len_data)
                    # print(data[y_i].skew())
                    # print(data[y_i].kurt())
                    res_dict[f'deviation{cla_i}'].append(len(data[data[y_i]>0]) / len_data)
                    res_dict[f'skew{cla_i}'].append(data[y_i].skew())
                    res_dict[f'kurt{cla_i}'].append(data[y_i].kurt())
                
                if len(real_cla_li) != cla:
                    cla_li = [i for i in range(cla)]
                    [cla_li.remove(i) for i in real_cla_li]
                    for i in cla_li:
                        res_dict[f'deviation{i}'].append(0)
                        res_dict[f'skew{i}'].append(0)
                        res_dict[f'kurt{i}'].append(0)

                if cla != np.max(self.index_class):  # 把少分类的补齐
                    for i in range(cla, np.max(self.index_class)):
                        res_dict[f'deviation{i}'].append(0)
                        res_dict[f'skew{i}'].append(0)
                        res_dict[f'kurt{i}'].append(0)

                if len(res_dict[f'deviation{0}']) != len(res_dict[f'deviation{4}']):
                    print(i, y_i)
                    print('index', len(res_dict['index']))
                    print('index_n', len(res_dict['index_n']))
                    print('interval', len(res_dict['interval']))
                    print('class', len(res_dict['class']))
                    print('y_name', len(res_dict['y_name']))
                    print('deviation0', len(res_dict['deviation0']))
                    print('deviation1', len(res_dict['deviation1']))
                    print('deviation2', len(res_dict['deviation2']))
                    print('deviation3', len(res_dict['deviation3']))
                    print('deviation4', len(res_dict['deviation4']))
                    print('skew0', len(res_dict['skew0']))
                    print('skew1', len(res_dict['skew1']))
                    print('skew2', len(res_dict['skew2']))
                    print('skew3', len(res_dict['skew3']))
                    print('skew4', len(res_dict['skew4']))
                    print('kurt0', len(res_dict['kurt0']))
                    print('kurt1', len(res_dict['kurt1']))
                    print('kurt2', len(res_dict['kurt2']))
                    print('kurt3', len(res_dict['kurt3']))
                    print('kurt4', len(res_dict['kurt4']))
                    print(res_dict['deviation4'])
                    print('------------')
                    # print(res_dict['deviation3'])
                    input()
        
        return res_dict
    
    def hist_res2(self, df0: pd.DataFrame, index, interval, adf_res, res_dict, symbol):
        '''计算分布情况'''
        df_params_sy, df_params_all = self.load_df_params(symbol)
        df_train = df0[df0['datetime'].dt.date < self.traindate.date()]     
        index_n = index.split('_')[-1]
        # real_index = index[:-len(index_n)-1]
        real_index = index.replace(f'_{index_n}', '')
        q = df_train[index].copy()
        df_index = df_params_sy[df_params_sy['index_name']==index]
        df_index_all = df_params_all[df_params_all['index_name']==index]
        if len(df_index_all['index_max']):
            if df_index_all['zero_mean'].iloc[0] == -1:
                range_max = df_index['index_max'].iloc[0]
                range_min = -range_max
            else:
                range_max, range_min = df_index['index_max'].iloc[0], df_index['index_min'].iloc[0]
        else:
            print(index, 'not in range')
            return res_dict

        for cla in self.index_class:    # y的分类
            step = (range_max-range_min) / cla
            df = df_train.copy()
            df['quantile'] = np.nan

            for i in range(0, cla):  # 对指标进行分类
                q_min, q_max = range_min+i*step, range_min+(i+1)*step
                df['quantile'] = np.where((q>=q_min) & (q<=q_max), i, df['quantile'])

                # if index == 'atrzc_11':
                #     print(index, q_max, q_min, range_min, range_max, q.max(), q.min(), len(df['quantile'].value_counts()))
                #     input()
            
            real_cla_li = df['quantile'].value_counts().index.to_list()
            
            if len(real_cla_li) == 0:
                print('overlap:', index)
                continue

            df.dropna(inplace=True)
            # print('quantile', df['quantile'].unique())
            
            for y_i in self.y_list:
                adf_n = 1 if index in adf_res else 0
                res_dict['smooth'].append(adf_n)
                res_dict['columns'].append(index), res_dict['corr'].append(np.sign(df[y_i].corr(q)))
                res_dict['index'].append(real_index), res_dict['index_n'].append(index_n)
                res_dict['interval'].append(interval), res_dict['class'].append(cla)
                res_dict['y_name'].append(y_i)

                for data in df.groupby('quantile'):  # 计算每个分类对应的y的分布
                    cla_i = int(list(data)[0])    
                    data = list(data)[1]    
                    len_data = len(data)
                    # print(data)
                    # print(len(data[data[y_i]>0]) / len_data)
                    # print(data[y_i].skew())
                    # print(data[y_i].kurt())
                    res_dict[f'deviation{cla_i}'].append(len(data[data[y_i]>0]) / len_data)
                    res_dict[f'skew{cla_i}'].append(data[y_i].skew())
                    res_dict[f'kurt{cla_i}'].append(data[y_i].kurt())
                
                if len(real_cla_li) != cla:
                    cla_li = [i for i in range(cla)]
                    [cla_li.remove(i) for i in real_cla_li]
                    for i in cla_li:
                        res_dict[f'deviation{i}'].append(0)
                        res_dict[f'skew{i}'].append(0)
                        res_dict[f'kurt{i}'].append(0)

                if cla != np.max(self.index_class):  # 把少分类的补齐
                    for i in range(cla, np.max(self.index_class)):
                        res_dict[f'deviation{i}'].append(0)
                        res_dict[f'skew{i}'].append(0)
                        res_dict[f'kurt{i}'].append(0)

                if len(res_dict[f'deviation{0}']) != len(res_dict[f'deviation{4}']):
                    print(i, y_i)
                    print('index', len(res_dict['index']))
                    print('index_n', len(res_dict['index_n']))
                    print('interval', len(res_dict['interval']))
                    print('class', len(res_dict['class']))
                    print('y_name', len(res_dict['y_name']))
                    print('deviation0', len(res_dict['deviation0']))
                    print('deviation1', len(res_dict['deviation1']))
                    print('deviation2', len(res_dict['deviation2']))
                    print('deviation3', len(res_dict['deviation3']))
                    print('deviation4', len(res_dict['deviation4']))
                    print('skew0', len(res_dict['skew0']))
                    print('skew1', len(res_dict['skew1']))
                    print('skew2', len(res_dict['skew2']))
                    print('skew3', len(res_dict['skew3']))
                    print('skew4', len(res_dict['skew4']))
                    print('kurt0', len(res_dict['kurt0']))
                    print('kurt1', len(res_dict['kurt1']))
                    print('kurt2', len(res_dict['kurt2']))
                    print('kurt3', len(res_dict['kurt3']))
                    print('kurt4', len(res_dict['kurt4']))
                    print(res_dict['deviation4'])
                    print('------------')
                    # print(res_dict['deviation3'])
                    input()
        
        return res_dict
    
    def save_res(self, symbol, res_dict):
        '''保存结果'''
        makedir(self.save_res_pa)
        df = pd.DataFrame(res_dict)
        df.dropna(inplace=True)
        df['table_index'] = df['columns'] + df['class'].apply(lambda x: f'_{x}')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df.set_index('table_index', inplace=True)
        df.to_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        
    def save_index(self, df: pd.DataFrame, symbol, interval):
        '''保存指标'''
        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'
        makedir(save_pa)
        df.to_csv(f'{save_pa}{symbol}_{interval}m.csv', index=False)

    def statistic_skew_scope(self, pa=f'{pa_prefix}/datas/factors_analyze/'):
        '''查看每个品种偏度的最大最小值，区间范围'''
        pa_li = os.listdir(pa)
        res_dic = {'symbols': []}
        [res_dic.update({f'skew{i}_max': [], f'skew{i}_min': [], f'skew{i}_mean': []}) for i in range(5)]

        for pa_i in pa_li:
            df_i = pd.read_csv(f'{pa}{pa_i}')
            df_i = df_i[(df_i['interval']==60) & (df_i['smooth']==1)]
            res_dic['symbols'].append(pa_i)
            for j in range(5):
                res_dic[f'skew{j}_max'].append(df_i[f'skew{j}'].max())
                res_dic[f'skew{j}_min'].append(df_i[f'skew{j}'].min())
                res_dic[f'skew{j}_mean'].append(df_i[f'skew{j}'].mean())
        df_res = pd.DataFrame(res_dic)
        df_res.to_csv(f'{pa}df_skew_scope.csv', index=False)
        print('statistic_skew_scope done')
        return df_res

    def skew_distribute_statistic(self, pa=f'{pa_prefix}/datas/factors_analyze/', save_pa=f'{pa_prefix}/datas/factors_skew_distribution/'):
        '''对因子的偏度的分布绘制统计图'''
        pa_li = filter_str('res.csv', os.listdir(pa), is_list=1)
        for pa_i in pa_li[20:]:
            df = pd.read_csv(f'{pa}{pa_i}')
            df = df[df['smooth']==1]
            save_name = pa_i.split('.')[0]
            save_pa_i = makedir(f'{save_pa}{save_name}/')
            for i in [3, 4, 5]:
                df_i = df[df['class']==i]
                [m_plot_one_hist(df_i[f'skew{j}'], f'{save_name}_{i}_{j}_'+str(df_i[f'skew{j}'].mean()), save_pa_i) 
                    for j in range(i)]
            print(save_pa_i, 'done.')

    def symbol_factor(self, symbol):
        '''获取一个品种不同周期的指标'''
        res_dict = self.init_dict()
        for interval in self.interval_li:   
            df_all = self.interval_factor(symbol, interval) # 计算指标
            save_pa = makedir(f'{pa_prefix}/datas/data_index/{symbol}/')
            df_all_pa = f'{save_pa}{symbol}_{interval}m_raw.csv'
            df_all.to_csv(df_all_pa, index=False)
            df_adj, res_dict, df_params = self.factors_analyze(df_all, interval, symbol, res_dict)  # 指标进行分析
            self.merge_stationary_params(df_params, res_dict, symbol, interval)
            self.save_index(df_adj, symbol, interval)
        self.save_res(symbol, res_dict)       # 保存指标分析结果
        return {'symbol': symbol, 'df': df_adj}

    def symbol_factor_copy(self, symbol):
        '''获取一个品种不同周期的指标'''
        for interval in self.interval_li:   
            df_all = self.interval_factor(symbol, interval) # 计算指标
            # df_all.to_csv(f'df_all_{interval}.csv')
            # pd.DataFrame(self.index_name).to_csv(f'index_name_{interval}.csv')
            df_all.fillna(method='ffill', inplace=True)
            df_all.dropna(axis=1, inplace=True)
            self.save_index(df_all, symbol, interval)

    def multi_symbol_factor_class(self, max_workers=3, is_save=0, futures_name='all'):
        '''多进程获取所有品种不同周期的指标'''
        interval = 60
        self.hist_res_method = 0   # 计算收益分布的方法
        self.need_del_stationary = 0
        # self.sy_li = self.syinfo.get_futures_li(futures_name)  
        self.sy_li = self.syinfo.get_futures_li('no_metals') 
        # self.sy_li = ['AP', 'RB', 'SF'] 
        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # max_workers=10
            res = executor.map(self.symbol_factor, self.sy_li)
        # exit()
        print('ProcessPoolExecutor done')
        df_index_li = []
        # for dic_i in res:
        #     print(dic_i['df'])
        #     print(dic_i['df'].columns)
        #     input()
        [df_index_li.append(dic_i['df']) for dic_i in res]
        df_index_all = pd.concat(df_index_li)  
        if is_save:
            pa = makedir(f'{pa_prefix}/datas/data_index/{futures_name}/')
            df_index_all.to_csv(f'{pa}df_index_{futures_name}.csv')

        res_dict = self.init_dict()
        df_adj, res_dict, df_params = self.factors_analyze(df_index_all, interval, futures_name, res_dict)  # 指标进行分析
        self.save_index(df_adj, futures_name, interval)
        self.merge_stationary_params(df_params, res_dict, futures_name, interval)
        self.process_df_params_all()
        self.save_res(futures_name, res_dict)       # 保存指标分析结果
        return df_adj

    def merge_stationary_params(self, df_params, res_dict, symbol, interval):
        '''合并参数表和统计结果表'''
        df_res = pd.DataFrame(res_dict)
        df_res = df_res[df_res['class']==3]
        df_res.rename(columns={'columns': 'index_name'}, inplace=True)
        df_res = df_res[['index_name', 'smooth']]
        df_merge = pd.merge(df_params, df_res, left_on='index_name', right_on='index_name', how='outer')
        df_merge.to_csv(f'{pa_prefix}/datas/data_index/{symbol}/{symbol}_{interval}m_params_merge.csv', index=False)
        print(f'{symbol}_{interval}m_params_merge done')
        return df_merge

    def process_df_params_all(self):
        '''处理参数范围'''
        pa = f'{pa_prefix}/datas/data_index/all/'
        df = pd.read_csv(f'{pa}all_60m_params_merge.csv')
        df = df[df['smooth']==1]
        df['zero_mean'] = np.sign(df['index_max']*df['index_min'])
        def process_x(x):
            if x['zero_mean'] == -1:
                if -0.01 < x['index_min'] < 0:
                    x['index_min'], x['zero_mean'] = 0, 0
            if x['zero_mean'] == -1:
                x['range_max'] = max(abs(x['index_min']), abs(x['index_max']))
                x['range_min'] = -x['range_max']
            else:
                x['range_max'] = max(x['index_min'], x['index_max'])
                x['range_min'] = min(x['index_min'], x['index_max'])
            return x

        df['range_max'], df['range_min'] = np.nan, np.nan     
        df = df.apply(process_x, axis=1)
        df.to_csv(f'{pa}all_60m_params_merge_adj.csv', index=False)
    
    def test_params_all(self):
        '''测试参数准确性'''
        pa = f'{pa_prefix}/datas/data_index/'
        sy_li = os.listdir(f'{pa}')
        test_val = 'skew30'
        sy_li.remove('all')
        df_all = pd.read_csv(f'{pa}all/all_60m_params.csv')
        dic = {'adx_11': [], 'atr_16': [], 'minus_di_21':[], 'stochf_11': []}
        col_li = list(dic.keys())
        for sy in sy_li:
            df_i = pd.read_csv(f'{pa}{sy}/{sy}_60m_params.csv')
            # df_i = df[df['index_name'].isin(col_li)]
            for col in col_li:
                dic[col].append(df_i[df_i['index_name']==col][test_val].iloc[0])
        for col in col_li:
            print(col, np.min(dic[col]), df_all[df_all['index_name']==col][test_val].iloc[0])
    
    def test_params_max_min_all(self):
        '''统计每个因子最大最小值'''
        pa = f'{pa_prefix}/datas/data_index/'
        sy_li = os.listdir(f'{pa}')
        sy_li.remove('all')
        df_all = pd.read_csv(f'{pa}all/all_60m.csv')
        df_all.drop(columns=self.columns_original, inplace=True)
        col_li = df_all.columns.to_list()
        res_dic = {}
        [res_dic.update({sy: []}) for sy in sy_li]
        index_li = []
        for col in col_li: 
            index_li.append(f'{col}_max')
            index_li.append(f'{col}_min')
        del df_all
        for sy in sy_li:
            df_i = pd.read_csv(f'{pa}{sy}/{sy}_60m.csv')
            df_i.drop(columns=self.columns_original, inplace=True)
            for col in col_li:
                try:
                    max_v, min_v = df_i[col].max(), df_i[col].min()
                except:
                    max_v, min_v = np.nan, np.nan
                res_dic[sy].append(max_v)
                res_dic[sy].append(min_v)
        df_res = pd.DataFrame(res_dic)
        df_res['index'] = index_li
        df_res.set_index('index', inplace=True)
        df_res.to_csv(f'{pa}all/df_max_min.csv')

    def symbol_factor_class(self, is_save=0, futures_name='all', factors_analyze=0):
        '''获取所有品种不同周期的指标'''
        df_index_li = []
        sy_li = self.syinfo.get_futures_li(futures_name)    

        for symbol in sy_li:
            print(symbol, 'begin.')
            for interval in self.interval_li: 
                save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'
                makedir(save_pa)
                df_all_pa = f'{save_pa}{symbol}_{interval}m_raw.csv'
                if len(self.columns_original) == 0:
                    df_all = self.interval_factor(symbol, interval) # 计算指标
                    df_all.to_csv(df_all_pa, index=False)
                else:
                    # try:
                    #     df_all = pd.read_csv(df_all_pa)  
                    # except:
                    df_all = self.interval_factor(symbol, interval) # 计算指标
                    df_all.to_csv(df_all_pa, index=False)
                        # pd.DataFrame(self.index_name).to_csv(f'index_name_{interval}.csv')
                if factors_analyze == 1:

                    df_all, res_dict, df_params = self.factors_analyze(df_all, interval, symbol, res_dict)  # 指标进行分析
                    self.save_index(df_all, symbol, interval)
                else:
                    df_all.fillna(method='ffill', inplace=True)
                    df_all.dropna(axis=1, inplace=True)
                df_index_li.append(df_all.copy())
                
            # if need_save_res:
            self.save_res(symbol, res_dict)       # 保存指标分析结果
            print(symbol, 'end.')
            
        df_index_all = pd.concat(df_index_li)  

        if is_save:
            pa = makedir(f'{pa_prefix}/datas/data_index/{futures_name}/')
            df_index_all.to_csv(f'{pa}df_index_{futures_name}.csv')
            
        res_dict = self.init_dict()
        df_adj, res_dict, df_params = self.factors_analyze(df_index_all, interval, futures_name)  # 指标进行分析

        self.save_index(df_adj, futures_name, interval)
        self.save_res(futures_name, res_dict)       # 保存指标分析结果
    
    def caculate_skew_mean_std(self, df0: pd.DataFrame, symbol, interval=60):
        '''计算偏度，均值和标准差'''
        # _, df, _ = self.process_df_before_analyze(df0)
        df = df0.copy()
        df = df[df['datetime'].dt.date < self.traindate.date()]
        col_li = df.columns.to_list()
        if 'datetime' in col_li: col_li = col_li[1:]
        if 'open' in col_li: col_li = col_li[6:]
        res_li = []
        for index_name in col_li:
            index_s = df[index_name]
            res_dic_i = {'index_name': index_name, 'mean': index_s.mean(), 'std': index_s.std(), 'max_min': (index_s.max()+index_s.min())/2, 
                         'index_max': index_s.max(), 'index_min': index_s.min()}
            for i in self.index_class:
                [res_dic_i.update({f'skew{i}{j}': index_s.quantile(k/100)}) for j, k in enumerate(range(0, 100, 100 // i))]
            res_li.append(deepcopy(res_dic_i))
        df_res = pd.DataFrame(res_li)
        save_pa = makedir(f'{pa_prefix}/datas/data_index/{symbol}/')
        df_res.to_csv(f'{save_pa}{symbol}_{interval}m_params.csv', index=False)
        return df_res

    def symbol_factor_all(self):
        '''按照板块计算因子'''
        for class_i in self.syinfo.futures_name_li:
            print('begin:', class_i)
            self.symbol_factor_class(is_save=1, futures_name=class_i, factors_analyze=0)

    def run_all_symbols_factor(self):
        '''跑所有品种的指标统计结果'''
        pa = f'{pa_prefix}/datas/data_1m/'
        symbol_li = os.listdir(pa)
        for symbol in symbol_li:
            symbol = symbol.split('.')[0]
            self.symbol_factor(symbol)

    def get_index_train_test_hist(self, pa=f'{pa_prefix}/datas/factors_analyze/all/all_60m.csv', split_date=None):
        '''获取因子在训练集和验证集上的分布图'''
        if split_date is None: split_date = self.traindate
        df = pd.read_csv(pa)
        col_li = df.columns.to_list()
        df['datetime'] = pd.to_datetime(df['datetime'])
        col_li = filter_str('_11', col_li, is_list=1)
        df_train = df[df['datetime']< split_date]
        df_test = df[df['datetime']>=split_date]
        save_pa = pa[:-len(pa.split('/')[-1])]
        df = df[col_li]
        for col in col_li:
            if col in self.columns_original:
                continue
            datas = [pd.DataFrame(df_train[col]), pd.DataFrame(df_test[col])]
            print(save_pa, col)
            m_plot_two_hist(datas, col, save_pa)
        return 

    def get_all_symbol_train_test_hist(self, pa=f'{pa_prefix}/datas/data_index/', interval=60):
        '''获取每个品种的各个因子分布图'''
        sy_li = os.listdir(pa)
        for sy in sy_li[30:32]:
            self.get_index_train_test_hist(pa=f'{pa}{sy}/{sy}_{interval}m.csv')
        return 

    def remove_corr_index(self, df: pd.DataFrame, col_li: list, corr_thread=0.98):
        '''去除相关性高的因子'''
        def count_corr(x):
            return len(x[x>corr_thread])
        count_corr_max = 30
        # col_li = df.columns.to_list()
        while count_corr_max > 1:       # 去除相关性高的因子
            df_all_corr = df[col_li].corr()
            corr_count = df_all_corr.apply(count_corr)
            count_corr_max = corr_count.max()
            index_name = corr_count.index[corr_count.argmax()]
            # del df_all[index_name]
            col_li.remove(index_name)
            # print(df_all_corr)

        return col_li

    def filter_skew_5(self, df_res, threadhold0, threadhold, i):
        '''筛选符合偏度阈值的因子'''
        if i == 3:
            # df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
            df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold) & ((df_res['skew0']*df_res[f'skew2']<0) | ((df_res['skew0']*df_res[f'skew2']>0) & ((df_res['skew0']/(df_res[f'skew2']+0.000001)<0.1) | (df_res['skew0']/(df_res[f'skew2']+0.000001)>10))))) | \
                            # ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                            ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold) & ((df_res['skew0']*df_res[f'skew2']<0) | ((df_res['skew0']*df_res[f'skew2']>0) & ((df_res['skew0']/(df_res[f'skew2']+0.000001)<0.1) | (df_res['skew0']/(df_res[f'skew2']+0.000001)>10)))))]
        
        elif i == 4:
            df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold) & ((df_res['skew0']*df_res[f'skew3']<0) | ((df_res['skew0']*df_res[f'skew3']>0) & ((df_res['skew0']/(df_res[f'skew3']+0.000001)<0.1) | (df_res['skew0']/(df_res[f'skew3']+0.000001)>10))))) | \
                            ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold) & ((df_res['skew1']*df_res[f'skew2']<0) | ((df_res['skew1']*df_res[f'skew2']>0) & ((df_res['skew1']/(df_res[f'skew2']+0.000001)<0.1) | (df_res['skew1']/(df_res[f'skew2']+0.000001)>10))))) | \
                            # ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                            ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold) & ((df_res['skew1']*df_res[f'skew2']<0) | ((df_res['skew1']*df_res[f'skew2']>0) & ((df_res['skew1']/(df_res[f'skew2']+0.000001)<0.1) | (df_res['skew1']/(df_res[f'skew2']+0.000001)>10))))) | \
                            ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold) & ((df_res['skew0']*df_res[f'skew3']<0) | ((df_res['skew0']*df_res[f'skew3']>0) & ((df_res['skew0']/(df_res[f'skew3']+0.000001)<0.1) | (df_res['skew0']/(df_res[f'skew3']+0.000001)>10)))))]

        elif i == 5:
            df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold) & ((df_res['skew0']*df_res[f'skew4']<0) | ((df_res['skew0']*df_res[f'skew4']>0) & ((df_res['skew0']/(df_res[f'skew4']+0.000001)<0.1) | (df_res['skew0']/(df_res[f'skew4']+0.000001)>10))))) | \
                            ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold) & ((df_res['skew1']*df_res[f'skew3']<0) | ((df_res['skew1']*df_res[f'skew3']>0) & ((df_res['skew1']/(df_res[f'skew3']+0.000001)<0.1) | (df_res['skew1']/(df_res[f'skew3']+0.000001)>10))))) | \
                            # ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                            ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold) & ((df_res['skew1']*df_res[f'skew3']<0) | ((df_res['skew1']*df_res[f'skew3']>0) & ((df_res['skew1']/(df_res[f'skew3']+0.000001)<0.1) | (df_res['skew1']/(df_res[f'skew3']+0.000001)>10))))) | \
                            ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold) & ((df_res['skew0']*df_res[f'skew4']<0) | ((df_res['skew0']*df_res[f'skew4']>0) & ((df_res['skew0']/(df_res[f'skew4']+0.000001)<0.1) | (df_res['skew0']/(df_res[f'skew4']+0.000001)>10)))))]
        return df_res

    def filter_factors(self, symbol, threadhold=0.6, max_num = 10, interval=5):
        '''过滤指标'''
        print('begin to filter factors...')
        suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
        method = 0 
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            df_li = []
            for i in self.index_class:  # 分类
                df_res = df_j[df_j['class']==i]
                if method == 0:
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[(df_res['skew0'].abs() > threadhold) | (df_res['skew1'].abs() > threadhold) | (df_res['skew2'].abs() > threadhold)]
                    elif i == 5:
                        df_res = df_res[(df_res['skew0'].abs() > threadhold) | (df_res['skew1'].abs() > threadhold) | \
                                        (df_res['skew2'].abs() > threadhold) | (df_res['skew3'].abs() > threadhold) | \
                                        (df_res['skew4'].abs() > threadhold)]

                elif method == 1:
                    threadhold = 0.3
                    if i == 3:
                        df_res = df_res[(df_res['skew0'].abs() < threadhold) & (df_res['skew1'].abs() < threadhold) & (df_res['skew2'].abs() < threadhold)]
                    elif i == 5:
                        df_res = df_res[(df_res['skew0'].abs() < threadhold) & (df_res['skew1'].abs() < threadhold) & \
                                        (df_res['skew2'].abs() < threadhold) & (df_res['skew3'].abs() < threadhold) & \
                                        (df_res['skew4'].abs() < threadhold)]
                
                elif method == 2:
                    if i == 3:
                        df_res = df_res[((df_res['deviation0']-0.5).abs() > threadhold) | \
                                        ((df_res['deviation1']-0.5).abs() > threadhold) | \
                                        ((df_res['deviation2']-0.5).abs() > threadhold)] 
                    elif i == 5:
                        df_res = df_res[((df_res['deviation0']-0.5).abs() > threadhold) | ((df_res['deviation1']-0.5).abs() > threadhold) | \
                                        ((df_res['deviation2']-0.5).abs() > threadhold) | ((df_res['deviation3']-0.5).abs() > threadhold) | \
                                        ((df_res['deviation4']-0.5).abs() > threadhold)]

                df_li.append(df_res.copy())

            df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
            if len(df_concat) == 0:
                print(j,'none-...')
                continue
            df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
            len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            print(len_index)
            print(j)
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            # df_concat['columns'].to_csv(f'{save_pa}{symbol}_{j}_index.csv', index=False)     # 将采样后的指标名称保存成csv
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold}_{max_num}_{stationary}_{j}.csv', index=False)
        print('done.')

    def filter_factors1(self, symbol, threadhold=0.6, max_num = 10, interval=5):
        '''过滤指标'''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
        method = 0 
        sample_i = -1
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        # df = df0[df0['interval']==interval]
        # for sample_i in sample_li:
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while len_index > max_num: # 23
                # threadhold += 0.05
                threadhold += 0.001
                df_li = []
                print('threadhold', threadhold)
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    if method == 0:
                        threadhold = threadhold
                        if i == 3:
                            df_res = df_res[(df_res['skew0'].abs() > threadhold) | (df_res['skew1'].abs() > threadhold) | (df_res['skew2'].abs() > threadhold)]
                        elif i == 5:
                            df_res = df_res[(df_res['skew0'].abs() > threadhold) | (df_res['skew1'].abs() > threadhold) | \
                                            (df_res['skew2'].abs() > threadhold) | (df_res['skew3'].abs() > threadhold) | \
                                            (df_res['skew4'].abs() > threadhold)]

                    elif method == 1:
                        threadhold = threadhold
                        if i == 3:
                            df_res = df_res[(df_res['skew0'].abs() < threadhold) & (df_res['skew1'].abs() < threadhold) & (df_res['skew2'].abs() < threadhold)]
                        elif i == 5:
                            df_res = df_res[(df_res['skew0'].abs() < threadhold) & (df_res['skew1'].abs() < threadhold) & \
                                            (df_res['skew2'].abs() < threadhold) & (df_res['skew3'].abs() < threadhold) & \
                                            (df_res['skew4'].abs() < threadhold)]
                    
                    elif method == 2:
                        if i == 3:
                            df_res = df_res[((df_res['deviation0']-0.5).abs() > threadhold) | \
                                            ((df_res['deviation1']-0.5).abs() > threadhold) | \
                                            ((df_res['deviation2']-0.5).abs() > threadhold)] 
                        elif i == 5:
                            df_res = df_res[((df_res['deviation0']-0.5).abs() > threadhold) | ((df_res['deviation1']-0.5).abs() > threadhold) | \
                                            ((df_res['deviation2']-0.5).abs() > threadhold) | ((df_res['deviation3']-0.5).abs() > threadhold) | \
                                            ((df_res['deviation4']-0.5).abs() > threadhold)]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                #     continue
                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
                print(len_index)
                print(j)
            print('before: ', df_concat.shape)
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            # df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.iloc[sample_i])     # 分组采样
            print('after', df_concat.shape)
            # df_concat['columns'].to_csv(f'{save_pa}{symbol}_{j}_index.csv', index=False)     # 将采样后的指标名称保存成csv
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_sample_{len_index}_{stationary}_{j}.csv', index=False)
        print('done.')

    def filter_factors2(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标'''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
        sample_i = -1
        threadhold = 4
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        # df = df0[df0['interval']==interval]
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                # threadhold += 0.05
                threadhold -= 0.001
                df_li = []
                # print('threadhold', threadhold)
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    # if method == 0:
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0
                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            # df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.iloc[sample_i])     # 分组采样
            # print('after', df_concat.shape)
            # df_concat['columns'].to_csv(f'{save_pa}{symbol}_{j}_index.csv', index=False)     # 将采样后的指标名称保存成csv
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors2_params(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标, 就算因子数量和阈值'''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4
        stationary = 1
        threadhold = 4
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return {}
                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            res_dic = {'symbol': [symbol], f'{threadhold0}_num': [len(df_concat)], f'{threadhold0}_thread': [threadhold]}
            return res_dic

    def filter_factors3(self, symbol, threadhold0=0.2, max_num = 7, interval=60, is_weight=1):
        '''过滤指标 
        1、skew0*skew2<0
        2、sort(std(skew0, skew2))
        '''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]

        def select_index(x: pd.DataFrame):
            '''通过std排序选择因子'''
            x = x.sort_values('std_skew', ascending=False)
            return x.iloc[0]

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            df_li = []

            for i in self.index_class:  # 分类
                df_res = df_j[df_j['class']==i]
                df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                df_res['std_skew'] = df_res[['skew0', f'skew{i-1}']].std(axis=1)
                df_res = df_res.groupby('index').apply(select_index)
                df_li.append(df_res.copy())
            
            df_concat = pd.concat(df_li).reset_index(drop=True)
            # df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            std_quantile = df_concat['std_skew'].quantile(threadhold0)  # 获取分位数
            df_concat = df_concat[df_concat['std_skew'] > std_quantile].sort_values('std_skew', ascending=True).iloc[:max_num]
            len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors4(self, symbol, threadhold0=0.2, max_num = 10, interval=60, is_weight=1):
        '''过滤指标 
        1、skew0*skew2<0
        2、abs(skew1)+abs(skew2)>thread 分4份
        3、abs(skew0)+abs(skew2)>thread 分3份
        4、abs(skew1)+abs(skew3)>thread 分5份
        5、abs(skew0)+abs(skew4)>thread 分5份
        '''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]

        def select_index(x: pd.DataFrame):
            '''通过std排序选择因子'''
            x = x.sort_values('std_skew', ascending=False)
            return x.iloc[0]

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            df_li = []

            for i in self.index_class:  # 分类
                df_res = df_j[df_j['class']==i]
                if i == 3:
                    df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                    df_res['skew_sum'] = df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs()
                    df_res['std_skew'] = df_res[['skew0', f'skew{i-1}']].std(axis=1)
                else:
                    df_res = df_res[(df_res['skew0']*df_res[f'skew{i-1}']<0) | (df_res['skew1']*df_res[f'skew{i-2}']<0)]
                    df_res['skew_sum1'] = np.where(df_res['skew0']*df_res[f'skew{i-1}']<0, df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs(), 0)
                    df_res['skew_sum2'] = np.where(df_res['skew1']*df_res[f'skew{i-2}']<0, df_res['skew1'].abs() + df_res[f'skew{i-2}'].abs(), 0)
                    df_res['skew_sum'] = df_res[['skew_sum1', 'skew_sum2']].max(axis=1)

                    df_res['std_skew'] = df_res[['skew0', f'skew{i-1}']].std(axis=1)
                    df_res['std_skew1'] = np.where(df_res['skew0']*df_res[f'skew{i-1}']<0, df_res[['skew0', f'skew{i-1}']].std(axis=1), 0)
                    df_res['std_skew2'] = np.where(df_res['skew1']*df_res[f'skew{i-2}']<0, df_res[['skew1', f'skew{i-2}']].std(axis=1), 0)
                    df_res['std_skew'] = df_res[['std_skew1', 'std_skew2']].max(axis=1)
                    
                df_res = df_res.groupby('index').apply(select_index)
                df_li.append(df_res.copy())
            
            df_concat = pd.concat(df_li).reset_index(drop=True)
            # df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            # std_quantile = df_concat['std_skew'].quantile(threadhold0)  # 获取分位数
            df_concat = df_concat[df_concat['skew_sum'] > threadhold0].sort_values('skew_sum', ascending=True).iloc[:max_num]
            len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors5(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标 2基础上加skew0*skew2<0'''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df_total = df_total[df_total['class']!=4]
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        stationary = 1
        threadhold = 4
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        # df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
        # df_all = df_all[df['columns'].to_list()]
        # df_all.to_csv('df1.csv')
        # exit()
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

            # 获取筛选之后的指标列并保存成csv
            len_index = len(df_concat)
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')
    
    def filter_factors6(self, symbol, threadhold0=0.2, max_num = 10, interval=60, is_weight=1):
        '''过滤指标 
        1、skew0*skew2<0
        2、abs(skew1)+abs(skew2)>thread 分4份
        3、abs(skew0)+abs(skew2)>thread 分3份
        4、abs(skew1)+abs(skew3)>thread 分5份
        5、abs(skew0)+abs(skew4)>thread 分5份
        '''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        stationary = 1
        threadhold = 4

        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    if i == 3:
                        df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                        df_res['skew_sum'] = df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs()
                        df_res['std_skew'] = df_res[['skew0', f'skew{i-1}']].std(axis=1)
                    else:
                        df_res = df_res[(df_res['skew0']*df_res[f'skew{i-1}']<0) | (df_res['skew1']*df_res[f'skew{i-2}']<0)]
                        df_res['skew_sum1'] = np.where(df_res['skew0']*df_res[f'skew{i-1}']<0, df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs(), 0)
                        df_res['skew_sum2'] = np.where(df_res['skew1']*df_res[f'skew{i-2}']<0, df_res['skew1'].abs() + df_res[f'skew{i-2}'].abs(), 0)
                        df_res['skew_sum'] = df_res[['skew_sum1', 'skew_sum2']].max(axis=1)

                        df_res['std_skew'] = df_res[['skew0', f'skew{i-1}']].std(axis=1)
                        df_res['std_skew1'] = np.where(df_res['skew0']*df_res[f'skew{i-1}']<0, df_res[['skew0', f'skew{i-1}']].std(axis=1), 0)
                        df_res['std_skew2'] = np.where(df_res['skew1']*df_res[f'skew{i-2}']<0, df_res[['skew1', f'skew{i-2}']].std(axis=1), 0)
                        df_res['std_skew'] = df_res[['std_skew1', 'std_skew2']].max(axis=1)
                        df_res.drop(columns=['skew_sum1', 'skew_sum2', 'std_skew1', 'std_skew2'], inplace=True)
                    
                    df_res = df_res[(df_res['skew_sum']>threadhold0) & (df_res['skew_sum']<threadhold)]
                    
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors7(self, symbol, threadhold0=0.2, max_num = 8, interval=60, is_weight=1):
        '''过滤指标 
        1、skew0*skew2<0
        2、a, b = abs(skew1), abs(skew2)
        3、a + b - |a-b| 进行排序,
        '''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]

        def select_index(x: pd.DataFrame):
            '''通过std排序选择因子'''
            x = x.sort_values('score', ascending=False)
            return x.iloc[0]

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            df_li = []

            for i in self.index_class:  # 分类
                df_res = df_j[df_j['class']==i]
                if i == 3:
                    df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                    df_res['score'] = df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs() - (df_res['skew0'].abs()-df_res[f'skew{i-1}'].abs()).abs()
                else:
                    df_res['score0'] = df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs() - (df_res['skew0'].abs()-df_res[f'skew{i-1}'].abs()).abs()
                    df_res['score1'] = df_res['skew1'].abs() + df_res[f'skew{i-2}'].abs() - (df_res['skew1'].abs()-df_res[f'skew{i-2}'].abs()).abs()
                    df_res['dir0'] = np.where(df_res['skew0']*df_res[f'skew{i-1}']<0, 1, 0)
                    df_res['dir1'] = np.where(df_res['skew1']*df_res[f'skew{i-2}']<0, 1, 0)
                    df_res['score0'] = df_res['score0']*df_res['dir0']
                    df_res['score1'] = df_res['score1']*df_res['dir1']
                    df_res['score'] = df_res[['score0', 'score1']].max(axis=1)
                    df_res.drop(columns=['score0', 'score1', 'dir0', 'dir1'], inplace=True) 
                    
                df_res = df_res.groupby('index').apply(select_index)
                df_li.append(df_res.copy())
            
            df_concat = pd.concat(df_li).reset_index(drop=True)
            # df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            std_quantile = df_concat['score'].quantile(threadhold0)  # 获取分位数
            df_concat = df_concat[df_concat['score'] > std_quantile].sort_values('score', ascending=True).iloc[:max_num]
            len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors8(self, symbol, threadhold0, thread_li):
        '''过滤指标 2基础上加skew0*skew2<0 固定因子数量'''
        print(symbol, 'begin to filter factors...')
        interval=60; is_weight=1
        threadhold_li = [0.7, 0.9, 1.1, 1.3, 1.5]
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        df_params = pd.read_csv(f'{pa_prefix}/datas/factors_skew_distribution/df_factor_2_params.csv')
        thread_i = threadhold_li[thread_li.index(threadhold0)]
        max_num = df_params[df_params['symbol']==symbol][f'{thread_i}_num'].iloc[0]
        if max_num == 0:
            print(symbol, threadhold0, 'get 0 num...')
            return {}

        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        stationary = 1
        threadhold = 4
        reduce_thread = 0.001
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= reduce_thread
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return {'symbol': [symbol], 'thread': [threadhold0], 'max_num': [max_num], 'actual': [0]}

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1)))        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            print(symbol, threadhold0, '最大数量:实际数量', max_num, len(df_concat), threadhold)
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')
            return {'symbol': [symbol], 'thread': [threadhold0], 'max_num': [max_num], 'actual': [len(df_concat)]}

    def filter_factors9(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标, 2的基础上加上判断相关性，去掉相关性高的因子'''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'
        df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

        stationary = 1
        threadhold = 4
        
        def count_corr(x):
            return len(x[x>0.99])
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary) & (df0['class']==3)]
        df_adj = df.groupby('index', group_keys=False).apply(lambda x: x.sample(1))
        count_corr_max = 30
        col_li = df_adj['columns'].to_list()
        while count_corr_max > 1:       # 去除相关性高的因子
            df_all_corr = df_all[col_li].corr()
            corr_count = df_all_corr.apply(count_corr)
            count_corr_max = corr_count.max()
            index_name = corr_count.index[corr_count.argmax()]
            # del df_all[index_name]
            col_li.remove(index_name)
            # print(count_corr_max)

        df_adj = df_adj[df_adj['columns'].isin(col_li)]
        df = df[df['index'].isin(df_adj['index'].to_list())]
        # df = df0[df0['interval']==interval]

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                # threadhold += 0.05
                threadhold -= 0.001
                df_li = []
                # print('threadhold', threadhold)
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    # if method == 0:
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0
                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            # df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.iloc[sample_i])     # 分组采样
            # print('after', df_concat.shape)
            # df_concat['columns'].to_csv(f'{save_pa}{symbol}_{j}_index.csv', index=False)     # 将采样后的指标名称保存成csv
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors10(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标 2基础上加skew0*skew2<0 5基础上加去除相关性高的因子'''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df_total = df_total[df_total['class']!=4]
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4
        
        # df0.to_csv('df0.csv')
        # print('done')
        # input()

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'
        df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

        stationary = 1
        threadhold = 4
        
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        df_adj = df0[(df0['interval']==interval) & (df0['smooth']==stationary) & (df0['class']==3)]
        # df_adj = df_adj.groupby('index', group_keys=False).apply(lambda x: x.sample(1))
        col_li = df_adj['columns'].to_list()
        # try:
        #     col_li = list(pd.read_csv(f'{pa_prefix}/datas/data_index'))
        col_li = self.remove_corr_index(df_all, col_li, corr_thread=0.90)
        # corr_count.to_csv('corr_count.csv')
        print('left index:', len(col_li))
        df_adj = df_adj[df_adj['columns'].isin(col_li)]
        df = df[df['index'].isin(df_adj['index'].to_list())]
        # df.to_csv('df.csv')
        # exit()

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    df_res = self.filter_skew_5(df_res, threadhold0, threadhold, i)
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

            # 获取筛选之后的指标列并保存成csv
            len_index = len(df_concat)
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')

    def filter_factors11(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标, 2的基础上加上判断相关性，5 筛选因子后再去掉相关性高的因子'''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df_total = df_total[df_total['class']!=4]
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        stationary = 1
        threadhold = 4
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        # df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
        # df_all = df_all[df['columns'].to_list()]
        # df_all.to_csv('df1.csv')
        # exit()
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

            col_li = self.remove_corr_index(df_all, df_concat['columns'].to_list(), corr_thread=0.98)
            # 获取筛选之后的指标列并保存成csv
            len_index = len(col_li)

            df_all = df_all[['datetime', j]+col_li]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')
    
    def filter_factors12(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标 2基础上加skewi*skewj<0'''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df_total = df_total[df_total['class']!=4]
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        stationary = 1
        threadhold = 4
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        # df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
        # df_all = df_all[df['columns'].to_list()]
        # df_all.to_csv('df1.csv')
        # exit()
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    if i == 3:
                        df_res = df_res[df_res['skew0']*df_res[f'skew{i-1}']<0]
                    elif i == 5:
                        df_res = df_res[(df_res['skew0']*df_res[f'skew{i-1}']<0) | (df_res['skew1']*df_res[f'skew{i-2}']<0) | \
                                        (df_res['skew0']*df_res[f'skew{i-2}']<0) | (df_res['skew1']*df_res[f'skew{i-1}']<0)]

                    threadhold = threadhold
                    if i == 3:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold))]
                    elif i == 5:
                        df_res = df_res[((df_res['skew0'].abs() > threadhold0) & (df_res['skew0'].abs() <= threadhold)) | \
                                        ((df_res['skew1'].abs() > threadhold0) & (df_res['skew1'].abs() <= threadhold)) | \
                                        ((df_res['skew2'].abs() > threadhold0) & (df_res['skew2'].abs() <= threadhold)) | \
                                        ((df_res['skew3'].abs() > threadhold0) & (df_res['skew3'].abs() <= threadhold)) | \
                                        ((df_res['skew4'].abs() > threadhold0) & (df_res['skew4'].abs() <= threadhold))]
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

            # 获取筛选之后的指标列并保存成csv
            len_index = len(df_concat)
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')
    
    def filter_factors13(self, symbol, threadhold0=5, max_num = 10, interval=60, is_weight=1):
        '''过滤指标 2基础上加skew0*skew2<0 5加上偏度大于0时数量级不一样'''
        print('begin to filter factors...')
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df_total = df_total[df_total['class']!=4]
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        stationary = 1
        threadhold = 4
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]
        # df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
        # df_all = df_all[df['columns'].to_list()]
        # df_all.to_csv('df1.csv')
        # exit()
        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            
            len_index = 500
            while (len_index > max_num and threadhold > threadhold0): # 23
                threadhold -= 0.001
                df_li = []
                for i in self.index_class:  # 分类
                    df_res = df_j[df_j['class']==i]
                    df_res = self.filter_skew_5(df_res, threadhold0, threadhold, i)
                    df_li.append(df_res.copy())

                df_concat = pd.concat(df_li)    # 将3分类和5分类的表合在一起
                if len(df_concat) == 0:
                    print(j,'none-...')
                    return 0

                df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
                len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格

            # 获取筛选之后的指标列并保存成csv
            len_index = len(df_concat)
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')
    
    def filter_factors14(self, symbol, threadhold0=0.2, max_num = 8, interval=60, is_weight=1):
        '''过滤指标 
        1、skew0*skew2<0
        2、a, b = abs(skew1), abs(skew2)
        3、a + b - |a-b| 进行排序,
        4、7基础上加如果两偏度同号但数量级不一样也可以
        '''
        print('begin to filter factors...')
        # suffix = '' if self.interval_li[-1] != 60 else '_60m'
        df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{self.interval_li[-1]}_res.csv')
        # df0 = pd.read_csv(f'{self.save_res_pa}{symbol}{suffix}_res.csv')
        if is_weight:
            df_total = pd.read_csv(f'{self.save_res_pa}total{self.interval_li[-1]}_res.csv')
            df0 = df0[df0['table_index'].isin(df_total['table_index'])]
            df_total = df_total[df_total['table_index'].isin(df0['table_index'])]
            caculate_li = ['deviation0', 'skew0', 'kurt0', 'deviation1', 'skew1', 'kurt1', 'deviation2', 
                        'skew2', 'kurt2', 'deviation3', 'skew3', 'kurt3', 'deviation4', 'skew4', 'kurt4']
            df0.set_index('table_index', inplace=True)
            df_total.set_index('table_index', inplace=True)
            df0[caculate_li] = df0[caculate_li]*0.6 + df_total[caculate_li]*0.4

        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'

        # for need_and in range(1):
        #     for stationary in range(3):
        stationary = 1
                
        df = df0[(df0['interval']==interval) & (df0['smooth']==stationary)]

        def select_index(x: pd.DataFrame):
            '''通过std排序选择因子'''
            x = x.sort_values('score', ascending=False)
            return x.iloc[0]

        for j in self.y_list:    # y标签
            df_j = df[df['y_name']==j]
            df_li = []

            for i in self.index_class:  # 分类
                df_res = df_j[df_j['class']==i]
                if i == 3:
                    df_res['dir0'] = np.where(df_res['skew0']*df_res[f'skew{i-1}']<0, 1, 0)
                    df_res['dir0'] = np.where((df_res['skew0'].abs()/df_res[f'skew{i-1}'].abs()>10)|(df_res['skew0'].abs()/df_res[f'skew{i-1}'].abs()<0.1), 1, df_res['dir0'])
                    df_res['score0'] = df_res['skew0'].abs() + df_res[f'skew{i-1}'].abs() - (df_res['skew0'].abs()-df_res[f'skew{i-1}'].abs()).abs()
                    df_res['score'] = df_res['score0']*df_res['dir0']
                    df_res.drop(columns=['dir'], inplace=True) 
                    
                else:
                    for l in range(2):
                        l1 = l+1
                        df_res[f'score{l}'] = df_res[f'skew{l}'].abs() + df_res[f'skew{i-l1}'].abs() - (df_res[f'skew{l}'].abs()-df_res[f'skew{i-l1}'].abs()).abs()
                        df_res[f'dir{l}'] = np.where(df_res[f'skew{l}']*df_res[f'skew{i-l1}']<0, 1, 0)
                        df_res[f'dir{l}'] = np.where((df_res[f'skew{l}'].abs()/df_res[f'skew{i-l1}'].abs()>10)|(df_res[f'skew{l}'].abs()/df_res[f'skew{i-l1}'].abs()<0.1), 1, df_res[f'dir{l}'])
                        df_res[f'score{l}'] = df_res[f'score{l}']*df_res[f'dir{l}']

                    df_res['score'] = df_res[['score0', 'score1']].max(axis=1)
                    df_res.drop(columns=['score0', 'score1', 'dir0', 'dir1'], inplace=True) 

                    
                df_res = df_res.groupby('index').apply(select_index)
                df_li.append(df_res.copy())
            
            df_concat = pd.concat(df_li).reset_index(drop=True)
            # df_concat = df_concat.drop_duplicates(subset=['index', 'index_n'], keep='first')       # 去掉重复的指标
            df_concat = df_concat.groupby('index', group_keys=False).apply(lambda x: x.sample(1))     # 分组采样
            std_quantile = df_concat['score'].quantile(threadhold0)  # 获取分位数
            df_concat = df_concat[df_concat['score'] > std_quantile].sort_values('score', ascending=True).iloc[:max_num]
            len_index = len(df_concat['index'].unique())        # 计算当前指标的数量
            df_all = pd.read_csv(f'{save_pa}{symbol}_{interval}m.csv')      # 读取原始指标表格
            # 获取筛选之后的指标列并保存成csv
            df_all = df_all[['datetime', j]+df_concat['columns'].to_list()]
            df_all = df_all.rename(columns={j: 'y'})
            # df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{round(threadhold, 3)}_{sample_i}_{len_index}_{stationary}_{j}.csv', index=False)
            df_all.to_csv(f'{save_pa}{symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j}.csv', index=False)
            print(f'{symbol} {symbol}_{interval}m_{threadhold0}_sample_{len_index}_{stationary}_{j} done.')
    
    def test(self):
        pa = f'{pa_prefix}/datas/data_index/'
        li = os.listdir(pa)
        for i in li:
            li_i = os.listdir(f'{pa}{i}/')
            lp = filter_str('m.csv', li_i, is_list=0)
            df = pd.read_csv(f'{pa}{i}/{lp}')
            print(i, df.shape)

def run_factorprocess_all(threadhold_li, symbol_li, is_symbol_factor):
    interval = 60
    # max_num = 10  # 20 10
    # threadhold_li = [1.2, 1.4, 1.6, 1.8]
    s = FactorProcess(interval=[interval])
    if is_symbol_factor:
        for symbol in symbol_li:
            s.symbol_factor(symbol)
        print('symbol_factor is done.')
    
    for symbol in symbol_li:
        for threadhold0 in threadhold_li:
            s.filter_factors10(symbol, threadhold0)
            # s.filter_factors8(symbol, threadhold0, threadhold_li)
            # s.filter_factors7(symbol, interval=interval, threadhold0=threadhold0)
            # s.filter_factors9(symbol, interval=interval, threadhold0=threadhold0, is_weight=1)
    print('filter_factors2 is done.')

def run_factorprocess_2_params():
    interval = 60
    symbol_li = ['AP', 'AG', 'AL', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 
                'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 
                'SF', 'SN', 'SR', 'TA', 'V', 'Y', 'ZN']
    threadhold_li = [0.7, 0.9, 1.1, 1.3, 1.5]
    s = FactorProcess(interval=[interval])
    res_li = []
    for symbol in symbol_li:
        res_dic_sy = {}
        for threadhold0 in threadhold_li:
            res_dic_i = s.filter_factors2_params(symbol, interval=interval, threadhold0=threadhold0)
            res_dic_sy.update(res_dic_i)
        res_li.append(res_dic_sy.copy())
    df_concat = pd.concat([pd.DataFrame(i) for i in res_li])
    df_concat.to_csv(f'{pa_prefix}/datas/factors_skew_distribution/df_factor_2_params.csv', index=False)
    print('run_factorprocess_2_1 is done.')

def run_factorprocess_8_debug():
    interval = 60
    symbol_li = ['AP', 'AG', 'AL', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 
                'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 
                'SF', 'SN', 'SR', 'TA', 'V', 'Y', 'ZN']
    threadhold_li = [0.7, 0.9, 1.1, 1.3, 1.5]
    s = FactorProcess(interval=[interval])
    res_li = []
    for symbol in symbol_li:
        # sy_func = partial(s.filter_factors8, symbol=symbol)
        # with ProcessPoolExecutor(max_workers=5) as executor:  # max_workers=10
        #     results = executor.map(sy_func, threadhold_li)
        # df_sy = pd.concat([pd.DataFrame(i) for i in results])
        # res_li.append(df_sy)
    
        for threadhold0 in threadhold_li:
            res_dic_i = s.filter_factors8(threadhold0=threadhold0, symbol=symbol)
            res_li.append(res_dic_i.copy())
    df_concat = pd.concat([pd.DataFrame(i) for i in res_li], ignore_index=True)
    df_concat.to_csv(f'{pa_prefix}/datas/factors_skew_distribution/df_factor_8_debug.csv', index=False)
    print('run_factorprocess_8_debug is done.')

def run_factorprocess(symbol='ru', max_num=10, interval=60, threadhold0=1.3):  # 0.6
    s = FactorProcess(interval=[interval])
    # s.symbol_factor(symbol)
    # s.filter_factors1(symbol, max_num=max_num, interval=interval, threadhold=threadhold)
    s.filter_factors2(symbol, max_num=max_num, interval=interval, threadhold0=threadhold0)

def search_same_index():
    symbol_li = SymbolsInfo().symbol_li
    df_all = pd.DataFrame()
    for symbol in symbol_li:
        save_pa = f'{pa_prefix}/datas/data_index/{symbol}/'
        df_all_pa = f'{save_pa}{symbol}_60m_raw.csv'
        df = pd.read_csv(df_all_pa)
        df.fillna(method='ffill', inplace=True)
        df.dropna(axis=1, inplace=True)
        print(symbol, df.shape)
        # df = pd.read_csv(f'{pa_prefix}/datas/factors_analyze/{symbol}60_res.csv')
        # if len(df_all) == 0:
        #     df_all = df.copy()
        # else:
        #     df_all = df_all[df_all['table_index'].isin(df['table_index'])]
    # df_all.to_csv('df_all.csv', index=False)
    # print(df_all.shape)


def run():
    is_symbol_factor = 1
    interval = 60
    max_num = 10  # 20 10
    symbol_li = ['AP']
    # threadhold_li = [1.2, 1.4, 1.6, 1.8]
    threadhold_li = [0.7*1.5, 0.9*1.5, 1.1*1.5, 1.3*1.5, 1.5*1.5]
    s = FactorProcess(interval=[interval])
    if is_symbol_factor:
        for symbol in symbol_li:
            s.symbol_factor(symbol)
        print('symbol_factor is done.')
    
    for symbol in symbol_li:
        for threadhold0 in threadhold_li:
            s.filter_factors2(symbol, max_num=max_num, interval=interval, threadhold0=threadhold0)
    print('filter_factors2 is done.')

def run_debug():
    s = FactorProcess()
    # s.symbol_factor_all()
    # s.symbol_factor_class(is_save=1, futures_name='all', factors_analyze=1)
    s.multi_symbol_factor_class(8)
    # s.get_index_train_test_hist()
    # s.get_all_symbol_train_test_hist()
    # s.test_params_all2()
    # s.process_df_params_all()
    # s.test()
    # s.statistic_skew_scope()
    # s.filter_factors8(threadhold0=1.3, symbol='AP')
    # s.skew_distribute_statistic()

if __name__ == "__main__":
    run_debug()
    # run_factorprocess_2_params()
    # run_factorprocess_8_debug()
    # search_same_index()
    


    
