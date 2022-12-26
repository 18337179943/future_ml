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
__Author__ = 'ZCXY'
from datas_process.m_futures_factors import MainconInfo
from datetime import timedelta, datetime
from m_base import filter_str
import joblib
from datas_process.m_futures_factors import *
from datas_process.m_datas_process import BaseDataProcess 
from simulation.simulation_backtester import SimulationBackTester
from simulation import BaseStrategy, SimulationStrategy
from datas_process.m_datas_process import *

'''做到caculate_index计算不同的指标'''

class MLTest():
    '''
    1、获取k线数据
    2、获取和计算技术指标
    3、对数据进行处理和保存对应参数, 包括均值和方差
    4、获取不同品种对应的模型
    5、获取模型的预测结果
    6、对预测结果进行回测
    7、对回测结果画图和保存
    '''
    def __init__(self, load_pa=f'{pa_prefix}/datas/ml_result/symbol_result_adj/params/', is_params_save=1, enddate=datetime(2022, 12, 14)) -> None:
        self.load_pa = load_pa
        self.symbol_res_pa_li = os.listdir(load_pa) # 获取每个品种的文件路径
        self.mainconinfo = MainconInfo()
        self.is_rq = 0 if 'datas_sc' in load_pa.split('/') else 1
        if self.is_rq == 0: self.mainconinfo.set_df_maincon()  # 用于datas_sc
        self.is_params_save = is_params_save
        self.startdate, self.enddate = datetime(2016, 1, 1), enddate   # datetime(2016, 1, 1), datetime(2020, 11, 20) datetime(2020, 12, 1), datetime(2021, 12, 31)
        self.train_date, self.val_date, self.test_date = datetime(2016, 1, 1), datetime(2019, 5, 1), datetime(2020, 5, 1)
        self.bt = SimulationBackTester(startdate=self.startdate, enddate=self.enddate, 
                                        strategy_class=SimulationStrategy)
        # self.symbols_li = ['AP', 'FG', 'HC', 'JD', 'JM', 'L', 'M', 'OI', 'P', 'RM', 'RU', 'V', 'sn', 'pp']
        self.symbols_li = ['CF', 'JD', 'M', 'P', 'V', 'ZN', 'OI', 'PP', 'RB', 'RU',
                           'SN', 'FG', 'Y', 'MA', 'RM']
        # self.symbols_li = ['CF', 'M', 'P', 'V', 'ZN', 'OI', 'RB', 'RU',
        #                     'FG', 'Y', 'MA']

        self.symbols_li = ['CF', 'JD', 'M', 'V', 'ZN', 'OI', 'PP', 'RB', 
                           'FG', 'RM', 'L', 'SF', 'Y', 'RU'] # adj

        self.symbols_li = ['AP', 'JD', 'M', 'V', 'OI', 'PP', 'RB', 
                           'FG', 'RM', 'L', 'SF', 'Y', 'RU', 'C'] # adj1

        self.symbols_li = ['AP', 'JD', 'M', 'V', 'OI', 'PP', 'RB', 
                           'FG', 'RM', 'SF', 'Y', 'RU'] # adj1

        self.symbols_li = self.get_symbol()


    def get_symbol(self):
        '''获取品种名称'''
        return [pa.split('_')[1] for pa in os.listdir(self.load_pa)]        

    def compare_mean_std_all(self):
        '''均值和标准差在验证集和测试集上进行对比'''
        for symbol in self.symbols_li:
            self.compare_mean_std(symbol)

    def compare_mean_std(self, symbol, save_pa='./datas/err_analyze/'):
        makedir(save_pa)
        bdp = BaseDataProcess()
        df_index = self.generate_y_pred(symbol, need_df_index=1)
        df_train = df_index[df_index.index < self.val_date]
        df_test = df_index[(self.val_date < df_index.index) & (df_index.index < datetime(2020, 12, 1))]
        df_2021 = df_index[(datetime(2020, 12, 1) < df_index.index) & (df_index.index < datetime(2021, 12, 31))]
        df_2022 = df_index[datetime(2021, 12, 31) < df_index.index]
        df_dic = {'train': df_train, 'test': df_test, '2021': df_2021, '2022': df_2022}
        dm_series_li, mean_std_li = [], []
        for key, df in df_dic.items():
            dm_series = bdp.cal_dm_and_d1(df)  #去极值
            dm_series.index = [f'dm_{key}', f'dm1{key}']
            dic_normal = bdp.cal_stats(df)  # 标准化
            df_mean_std = pd.DataFrame()
            df_mean_std[f'mean_{key}'] = dic_normal['mean']
            df_mean_std[f'std_{key}'] = dic_normal['std']
            dm_series_li.append(dm_series.copy())
            mean_std_li.append(df_mean_std.copy())
        
        df_dm = pd.concat(dm_series_li)
        df_mean_std = pd.concat(mean_std_li, axis=1)
        df_dm.to_csv(f'{save_pa}{symbol}_dm.csv')
        df_mean_std.to_csv(f'{save_pa}{symbol}_mean_std.csv')
        return df_dm, df_mean_std

    def generate_y_pred(self, symbol, interval='60', need_df_index=0):
        '''主函数'''
        print(symbol)
        self.index_name_n = []
        if self.is_rq == 0:
            load_pa = f'{pa_prefix}/datas_sc/data_{interval}m/{symbol}/'  # K线存放路径
        else:
            load_pa = f'{pa_prefix}/datas/data_{interval}m/{symbol}/'  # K线存放路径
        symbol_pa = filter_str(f'_{symbol}_', self.symbol_res_pa_li)    # 品种结果存放路径
        y_thread = eval(symbol_pa.split('_')[0])
        symbol_pa = f'{self.load_pa}{symbol_pa}/'
        symbol_pa_li = os.listdir(symbol_pa)     # 品种结果相关文件路径
        model_pa = symbol_pa + filter_str('model', symbol_pa_li)    # 模型
        index_pa = symbol_pa + filter_str('_adj', symbol_pa_li)     # 技术指标相关
        pred_save_pa = f'{symbol_pa}{symbol}_final_test_pred.csv'
        df_index_save_pa = f'{symbol_pa}{symbol}_final_df_index.csv'

        try:
            dm_series_pa = filter_str('dm_series', symbol_pa_li)    # 极值参数
            dic_normal_pa = filter_str('dic_normal', symbol_pa_li)  # 标准化参数
            dic_pa = {'dm_series_pa': f'{symbol_pa}{dm_series_pa}', 'dic_normal_pa': f'{symbol_pa}{dic_normal_pa}', 
                      'save_pa': df_index_save_pa}
        except:
            dic_pa = {'dm_series_pa': f'{symbol_pa}{symbol}_dm_series.pkl', 'dic_normal_pa': f'{symbol_pa}{symbol}_dic_normal.pkl',
                      'save_pa': df_index_save_pa}
        
        df_index_info = self.get_index_name(index_pa)       # 获取技术指标和对应的参数
        index_columns = df_index_info['columns'].to_list()
        model = self.load_model(model_pa)   # 获取模型
        df_contracts = self.mainconinfo.get_symbol_df_maincon(symbol, 
                        self.startdate, self.enddate, delay=15, cut=0)   # 获取主力合约列表和主力合约时间段 #  15
        # print(df_contracts.head(30))
        # input()

        df_all = []
        for i in range(len(df_contracts)):      # 品种的合约
            contract, startdate, enddate, symbol = df_contracts.iloc[i]
            # print(contract, startdate, enddate)
            df = pd.read_csv(f'{load_pa}{contract}.csv')    # 获取品种合约k线
            df['datetime'] = pd.to_datetime(df['datetime']) 
            if (enddate - df['datetime'].iloc[0]).days < 10:
                print(contract, startdate, enddate, symbol, df['datetime'].iloc[0], (enddate - df['datetime'].iloc[0]).days)
                print('del ', contract)
                input()
                continue
            df = df[(df['datetime'].dt.date >= startdate) & (df['datetime'].dt.date <= enddate)]    # 获取合约主力时间段
            df = self.get_y_values(df)  # 获取y值
            df = self.caculate_index(df_index_info, df)    # 计算指标
            df = df[(df['datetime'] >= startdate+timedelta(15)) & (df['datetime'] <= enddate)]
            # df = df[(df['datetime'].dt.date >= startdate) & (df['datetime'].dt.date <= enddate)]
            df_all.append(df)
        df_all = pd.concat(df_all)  # 合约拼接
        df_all.fillna(method='ffill', inplace=True)
        df_all = df_all.drop_duplicates(subset=['datetime'], keep='first')  # 去重
        df_all.set_index('datetime', inplace=True)
        df_all.dropna(inplace=True)
        df_index = df_all[index_columns]    # 筛选需要的指标
        if need_df_index:
            return df_index

        df_index = self.index_process(df_index, **dic_pa) # 数据进行处理
        y = self.get_y_label(df_all['y'], y_thread)     # 获取y值的标签
        y_pred = self.get_model_pred_res(model, df_index, y, pred_save_pa)   # 获取模型的预测值
        y_pred.reset_index(inplace=True)
        return y_pred
    
    def main(self, symbol=None, params={}):
        '''回测'''
        if symbol is None:
            y_pred_li = self.generate_y_pred_total()
            self.run_backtest_combo(self.symbols_li, y_pred_li)
        else:
            y_pred = self.generate_y_pred(symbol)
            self.run_backtest(symbol, y_pred, params)

    def generate_y_pred_total(self, symbols_li=None):
        '''获取所有品种的预测值'''
        if symbols_li is None:
            symbols_li = self.symbols_li
        y_pred_li = []
        for symbol in self.symbols_li:
            y_pred_li.append(self.generate_y_pred(symbol))
        return y_pred_li

    def get_y_values(self, df_o):
        '''获取y值'''
        df = df_o.copy()
        df['y'] = df['close'].pct_change().shift(-1)
        df.dropna(inplace=True)
        return df

    def get_y_label(self, df: pd.DataFrame, y_thread):
        '''计算y值'''
        def classify(x):
            if x < -value_thread:
                return 0
            elif x > value_thread:
                return 2
            else:
                return 1

        if isinstance(y_thread, list):
            n = y_thread[0]
            k = y_thread[1]
            need_mean = y_thread[2]
            y_method = y_thread[3]
            y_std = df.rolling(n).std().shift(1)
            y = np.ones(len(y_std))
            y_mean = df.rolling(n).mean().shift(1) if need_mean else 0
            if y_method == 0:
                y = np.where((df>y_mean+k*y_std) & (df>0), 2, y)
                y = np.where((df<y_mean-k*y_std) & (df<0), 0, y)
            elif y_method == 1:
                y = np.where(df>abs(y_mean+k*y_std), 2, y)
                y = np.where(df<-abs(y_mean+k*y_std), 0, y)
        
        elif y_thread == 0:
            y = df.apply(np.sign)
        else:
            df_train = df[df.index<self.val_date]
            value_thread = df_train.abs().quantile(y_thread)
            y = df.apply(classify)
        return y

    def index_process(self, df: pd.DataFrame, dm_series_pa, dic_normal_pa, save_pa):
        '''获取均值和标准差, 对技术指标进行数据处理'''
        dpm = DataProcessML()
        df = dpm.process_factors_test(df, dm_series_pa, dic_normal_pa, save_pa, is_save=self.is_params_save)
        # bdp = BaseDataProcess()
        # try:
        #     dm_series = joblib.load(dm_series_pa)    # dataframe
        #     dic_normal = joblib.load(dic_normal_pa)  # dic
        # except:
        #     df_train = df[df.index<self.val_date]
        #     dm_series = bdp.cal_dm_and_d1(df_train) # #去极值
        #     dic_normal = bdp.cal_stats(df_train)  # 标准化
        #     joblib.dump(dm_series, dm_series_pa)
        #     joblib.dump(dic_normal, dic_normal_pa)
        
        # df = bdp.mad(df, dm_series)
        # df = bdp.znormal(df, dic_normal) 
        # if len(save_pa):
        #     df.to_csv(save_pa)
        return df            

    def get_index_name(self, pa):
        '''获取指标名称'''
        df_index_info = pd.read_csv(pa)
        df_index_info = self.revise_index_name(pa, df_index_info.copy())
        return df_index_info

    def revise_index_name(self, pa, df: pd.DataFrame):
        '''修正错误的因子名称'''
        index_adj = ['macd_', 'rocr_100_', 'stoch_', 'log10_']
        index_p = {'0': ['macd', 'ta_lib'], '1': ['rocr_100', 'ta_lib'], '2': ['stoch', 'ta_lib'], '3': ['log10', 'ta_lib']}
        df_res = df.copy()
        def detect_index(x):
            try:
                index_n = [(i in x)*1 for i in index_adj].index(1)
            except:
                index_n = -1
            return index_n

        df_res['index_r'] = list(map(detect_index, df['columns'].to_list()))
        index_li = df_res[df_res['index_r']!=-1].index.to_list()
        if len(index_li):
            for i in index_li:
                index_name = index_p[str(df_res['index_r'].iloc[i])][0]
                df['index_i'].iloc[i] = index_name
                df['index_name'].iloc[i] = index_name
                df['index_category'].iloc[i] = index_p[str(df_res['index_r'].iloc[i])][1]
            df.to_csv(pa, index=False)
        return df


    def caculate_index(self, df_index_info, df: pd.DataFrame):
        '''计算技术指标值'''
        def get_index(df: pd.DataFrame, index_func, name, n):
            '''获取指标'''
            func = getattr(index_func, name)
            params_n = func.__code__.co_argcount    # 判断指标需要输入的参数
            if params_n == 1:
                res = func()
            else:
                res = func(n)
            index_name = f'{name}_{n}'
            df[index_name] = res if type(res) is not tuple else res[0]    # 判断返回的参数
            return df

        index_func = FactorIndex(df)
        # print(df_index_info['index_name'].iloc[:])
        for i in range(len(df_index_info)):
            index_type = df_index_info['index_category'].iloc[i]    # 指标分类
            index_name = df_index_info['index_name'].iloc[i]    # 指标名称
            # print(index_name)
            index_n = df_index_info['win_n'].iloc[i]    # 指标参数

            if index_type == 'pandas_tb':
                df, index_name_li = index_func.pandas_ta(df, index_n, index_name)
                # print(index_name_li)
                # input()
            elif index_type == 'ta_lib':
                df = get_index(df, index_func, index_name, index_n)
            else:
                print('其他指标')
                pass
        return df
    
    def load_model(self, model_pa):
        '''加载模型'''
        model = joblib.load(model_pa)
        return model

    def get_model_pred_res(self, model, x, y=None, save_pa=None):
        '''获取模型预测的结果'''
        y_pred = pd.DataFrame(model.predict_proba(x))
        y_pred.columns = ['decline', 'zero', 'rise']
        y_pred.index = x.index
        y_pred['y_pred'] = model.predict(x)
        if y is not None:
            y_pred['y_real'] = y
        # print(y_pred['y_real'].value_counts())
        if save_pa is not None:
            y_pred.to_csv(save_pa)
        return y_pred

    def run_backtest(self, symbol, y_pred, params={}):
        '''对模型结果进行回测'''
        self.bt.signal_analyze(symbol, y_pred, self.train_date, self.val_date, self.test_date, self.enddate, params=params)
    
    def run_backtest_combo(self, symbol_li, y_pred_li, params={}):
        self.bt.signal_analyze_total(symbol_li, y_pred_li, self.train_date, self.val_date, self.test_date, self.enddate, params=params)


class ModelCombo():
    def __init__(self) -> None:
        pass
    
    def model_init(self, pa_dic):
        '''初始化模型'''
        self.model = joblib.load(pa_dic['model_pa'])
        self.dm_series = joblib.load(pa_dic['dm_series_pa'])    # dataframe
        self.dic_normal = joblib.load(pa_dic['dic_normal_pa'])  # dic
        self.df_index_info = pd.read_csv(pa_dic['index_pa'])   # 需要的指标信息
        self.index_columns = self.df_index_info['columns'].to_list()  # 指标名称

    def get_index(self, df: pd.DataFrame, index_func, name, n):
        '''获取指标'''
        func = getattr(index_func, name)
        params_n = func.__code__.co_argcount    # 判断指标需要输入的参数
        if params_n == 1:
            res = func()
        else:
            res = func(n)
        index_name = f'{name}_{n}'
        df[index_name] = res if type(res) is not tuple else res[0]    # 判断返回的参数
        return df

    def transfrom_sig(self, sig):
        '''转换信号'''
        if sig == 0:
            sig = -1
        elif sig == 1:
            sig = 0
        else:
            sig = 1

    def mad(self, df, dm_series = None):
        '''
        中位数法去极值
        '''
        # print('整体去极值')

        def fun(series, dm_series):
            # print(dm_series.loc[:, series.name])
            dm1 = dm_series[series.name].loc['dm1']
            dm = dm_series[series.name].loc['dm']
            # 超过/小于 dm + 5dm1/dm - 5dm1 的修改为 dm + 5dm1/dm - 5dm1
            series[series > dm + 5 * dm1] = dm + 5 * dm1
            series[series < dm - 5 * dm1] = dm - 5 * dm1
            return series
        try:
            if dm_series == None:
                dm_series = self.cal_dm_and_d1(df)
        except:
            pass
        if len(df.columns) != len(dm_series.columns):
            print(len(df.columns), len(dm_series.columns), '参数不一致')
        # print(dm_series)
        df = df.apply(fun, args=(dm_series, ))
        return  df
        
    def znormal(self, df , stats = None):
        '''Standardization'''
        # print('整体归一化')
        try:
            if stats == None:
                stats = self.cal_stats(df)
        except:
            pass
        if len(df.columns) != len(stats['mean'].index):
            print(len(df.columns), len(stats['mean'].index), '参数不一致')

        result = (df - stats['mean'] )/ stats['std']
        return result

    def index_process(self, df: pd.DataFrame, dm_series, dic_normal):
        '''获取均值和标准差, 对技术指标进行数据处理'''
        df = self.mad(df, dm_series)
        df = self.znormal(df, dic_normal) 
        return df    
    
    def caculate_index(self, df_index_info, df: pd.DataFrame):
        '''计算技术指标值'''
        index_func = FactorIndex(df)
        # print(df_index_info['index_name'].iloc[:])
        for i in range(len(df_index_info)):
            index_type = df_index_info['index_category'].iloc[i]    # 指标分类
            index_name = df_index_info['index_name'].iloc[i]    # 指标名称
            # print(index_name)
            index_n = df_index_info['win_n'].iloc[i]    # 指标参数

            if index_type == 'pandas_tb':
                df, index_name_li = index_func.pandas_ta(df, index_n, index_name)
                # print(index_name_li)
                # input()
            elif index_type == 'ta_lib':
                df = self.get_index(df, index_func, index_name, index_n)
            else:
                print('其他指标')
                pass
        return df

    def model_predict(self, amn):
        '''模型预测'''
        df = pd.DataFrame()
        df['open'], df['high'], df['low'], df['close'], df['volume'], df['turnover'] = \
            amn.open, amn.high, amn.low, amn.close, amn.volume, amn.turnover

        df = self.caculate_index(self.df_index_info, df)  # 计算因子值
        df_index = df[self.index_columns]    # 筛选需要的指标
        df_index = self.index_process(df_index, self.dm_series, self.dic_normal)
        y_pred = pd.DataFrame()
        y_pred['y_pred'] = self.model.predict(df_index)  # 预测信号
        sig = self.transfrom_sig(y_pred['y_pred'].iloc[-1]) # 信号转换
        return sig


def run_mltest():
    mlt = MLTest()
    symbols_li = ['AP', 'FG', 'HC', 'JD', 'JM', 'L', 'M', 'OI', 'P', 'RB', 'RM', 'RU', 'V', 'sn', 'pp']
    # for i in symbols_li:
    #     print(i)
    mlt.main()

if __name__ == '__main__':
    # run_mltest()
    # mlt = MLTest(load_pa=f'{pa_prefix}/datas/ml_result/symbol_result_10_index/params/')
    # mlt.compare_mean_std_all()
    mainconinfo = MainconInfo()
    df_contracts = mainconinfo.get_symbol_df_maincon('SF', 
                        datetime(2016, 1, 1), datetime(2022, 12, 14), delay=15, cut=0)
    print(df_contracts)

    