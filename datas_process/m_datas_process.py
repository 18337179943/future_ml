from tkinter import SEL
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
import time
from datetime import time as dttime
from datetime import datetime, timedelta, time
from rx import interval
import scipy.stats as st
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.'
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
import joblib
from datas_process.nk_strategy import NkLine
# from nk_strategy import NkLine
from vnpy_ctastrategy.backtesting import BacktestingEngine
from m_base import *
from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import get_database
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from datas_process import m_futures_factors as mff, zigzag 
# import m_futures_factors as mff
import statsmodels.tsa.stattools as ts
import scipy.stats as st
from copy import deepcopy
import statsmodels.api as sm
from datas_process.m_futures_factors import FactorIndexStatistics
from datas_process.zigzag import *
import warnings
# from mainconinfo.maincon import run_change_maincon_to_rq
warnings.filterwarnings('ignore')
# import multiprocessing as mp


class BaseDataProcess:
    '''数据处理基类'''
    def normal_test(self, df, alpha = 0.05):
        '''
        正态分布的检测
        ----------
        df : 数据
        alpha : 显著度 The default is 0.05.
        -------
        '''
        df = df.dropna(axis = 1 , how = 'all')
        p_value = pd.Series( st.normaltest(df , axis = 0, nan_policy = 'omit').pvalue,
                               index = df.columns , name = 'pvalue' )
        return p_value[p_value < 0.05].index
    
    def select_adf_test_res(self, df):
        '''判断每列时间序列是否平稳'''
        adf_res = df.apply(self.adf_test)
        adf_res = pd.DataFrame(adf_res)
        adf_res = adf_res.reset_index()
        adf_res.columns = ['ind', 'is_smooth']
        adf_res = adf_res[adf_res['is_smooth']==True]
        return adf_res['ind'].to_list()

    def adf_test(self, series , alpha = 0.01):
        '''
        平稳序列的检测
        ----------
        series : 序列
        alpha : 显著度 The default is 0.01.
        -------
        '''
        try:
            # 原假设为存在单位根，
            if ts.adfuller(series , 1)[1] < alpha: return True 
            else: return False
        except:
            return False

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

    def cal_dm_and_d1(self, df):
        '''按列计算中位数去极值中的参数
        -----
        df：一般行为时间序列，列为因子
        返回：dataframe， 行为dm，dm1，列为原df列名'''
        def fun(series):
            dm = series.median()
            dm1 = np.abs(series - dm).median()
            return pd.Series([dm, dm1], index = ['dm', 'dm1'])
        return df.apply(fun)


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

    def cal_stats(self, df):
        return {'mean' : df.mean() , 'std' : df.std()} 

    def std_(self, df):
        '''
        3sigma去极值
        '''
        temp = df
        def fun(series):
            mean = series.mean()
            std = series.std()
            # 超过/小于 mean + 3sigma/mean - 3sigma 的修改为 mean  + 3sigma/mean  - 3sigma
            series[series > (mean + 3 * std)] = mean + 3 * std
            series[series < (mean - 3 * std)] = mean - 3 * std
            return series
        return temp.apply(fun)  
    
    def boxplot_(self, df):
        temp = df
        def fun(series): #百分位法
          iql = series.quantile(0.75) - series.quantile(0.25)
          series[series > (series.quantile(0.75) + 1.5 * iql)] = series.quantile(0.75) + 1.5 * iql
          series[series < (series.quantile(0.25) - 1.5 * iql)] = series.quantile(0.25) - 1.5 * iql
          return series
        return temp.apply(fun)   
        
    def percentile_(self, df):
        temp = df
        def fun(series,min = 0.10,max = 0.90): #百分位法
          series = series.sort_values()
          q = series.quantile([min,max])
          return np.clip(series,q.iloc[0],q.iloc[1])
        temp = temp.apply(fun,axis=1)
        return temp  

    def mean_normal_(self, df):
        '''Mean normalization'''
        temp = df
        result = (temp - temp.mean())/ (temp.max() - temp.min())#(x - mean)/(max - min)
        return result
    
    def min_max_(self, df):
        '''min-max normalization'''
        temp = df
        result = (temp.sub(temp.min(1) , 0)).div(temp.max(1) - temp.min(1), 0) #(x - min)/(max - min)
        return result
    
    def neutralization(self, factors, neu, factor_name= None):
        '''
        市值，行业中性化
        ----------
        factors : 因子值, 列为时间序列，行为股票
        industry : 行业表
        neu : 中性化的因子标的
        -------
        temp : 处理后的因子
        '''
        temp_f = factors
        start = time.time()
        print(time.time() - start , temp_f, factor_name)
        # mrkt = np.log(mrkt)
        #获取行业因子哑变量 （0，1）
        def fun(factors, mrkt):
            '''因子做为因变量对neu因子做线性回归'''
            x = mrkt.loc[factors.name , factors.index].to_frame() #只用市值
            y = factors
            factors = sm.OLS(y, x, missing='drop').fit().resid.reindex(factors.index)
            return factors
        temp_f = temp_f.apply(fun, args = (neu), axis = 1)
        print(time.time() - start )
        return  temp_f
      
    def dummy_neutralization(self, factors, dummy):
        start = time.time()
        print(time.time() - start , factors, factors)
        #check factors，industry，mrkt的size
        if (~factors.columns.isin(dummy.TICKER_SYMBOL)).sum() != 0:
            old_shape = factors.shape[1]
            factors = factors.loc[:,factors.columns.isin(dummy.TICKER_SYMBOL)]
            print('某些个股没有行业分类的数据，去除这些个股')
            print('总共剔除了{}个个股, 剩余股票数为{}'.format(old_shape - factors.shape[1], factors.shape[1]))
        # mrkt = np.log(mrkt)
        #获取行业因子哑变量 （0，1）
        dummies = pd.get_dummies(dummy,dummy_na=False)
        dummies.index = dummy.TICKER_SYMBOL
        def fun(factor, dummies):
            '''因子做为因变量对行业因子做线性回归'''
            x = factor.to_frame().join(dummies)
            x = x.iloc[:, -(len(dummy)):]
            y = factor
            factors = sm.OLS(y, x, missing='drop').fit().resid.reindex(factor.index)
            return factors
        factors = factors.apply(fun, args = (dummies), axis = 1)
        print(time.time() - start )
        return factors 

    def adf_singlefactor(self, df: pd.DataFrame, alpha = 0.01):
        '''
        单因子平稳序列检测
        ----------
        df : 单因子数据，
        alpha : 显著度
        ----------
        Returns: 判定为平稳序列的所有个股数据
        '''
        true = []
        df = df.dropna(axis = 1, how ='all') 
        for i in range(df.shape[1]):
            temp = df.iloc[:,i][df.iloc[:,i].first_valid_index():] #截取有效时间段
            if self.adf_Test(temp , alpha) == True: #单序列检测
                true.append(i)
        return df.columns[true]
    
    def fillna(self, df, method = 'ffill', value = None):
        '''
        单因子null的整理
        ----------
        df : 
                                         EPIBS  EgibsLong
            2010-01-04 000059.XSHE   1.028745  -0.087136
                       000096.XSHE        NaN        NaN
                       000159.XSHE        NaN        NaN
                       000301.XSHE   0.018974  -0.063350
                       000554.XSHE        NaN        NaN
                                          ...        ...
            2020-12-24 603353.XSHG  -0.382629  -0.239442
                       603619.XSHG        NaN        NaN
                       603727.XSHG  -0.320906   2.071929
                       603798.XSHG        NaN        NaN
                       603800.XSHG        NaN        NaN
        value : 替换值
        method : 替换方法  {'backfill', 'bfill', 'pad', 'ffill', None}, default None。定义了填充空值的方法， pad / ffill表示用前面行/列的值，填充当前行/列的空值， backfill / bfill表示用后面行/列的值，填充当前行/列的空值。
        limit : 最多填充前 limit 个空值
        -------
        result : 处理后的因子数据     '''
        # print('填充缺失值')
        if isinstance(value, int) or isinstance(value, pd.Series):
            return df.fillna(value)
        else:
            df.fillna(method)
    

    def check_if_new(self, series, days = 360):
        '''在每个时间截面剔除days以内上市的新股'''
        if (~series.isnull()).sum() == 0 or (series.isnull()).sum() == 0: 
            return series
        # 按ipo后三个月的时间点，当作股票的有效时间点，截取时间序列
        ipo_date = series.first_valid_index()
        valid_date = ipo_date + timedelta(days)
        return series[valid_date:]


    def to_multiindex(self, df):
        '''多索引格式转换
        -----
        df : 
            asset       000059.XSHE  000096.XSHE  ...  
            date                                  ...                          
            2010-01-04         12.0         11.0  ...       
            2010-01-11         18.0         10.0  ...        
            2010-01-18          9.0          7.0  ...         
            2010-01-25         14.0         17.0  ...   
        -----
        return :
                                    value
            date       asset             
            2010-01-04 000059.XSHE   12.0
                    000096.XSHE   11.0
                    000159.XSHE    4.0
                    000301.XSHE    3.0
                    000554.XSHE   18.0
                                ...
            2020-12-21 603353.XSHG   14.0
                    603619.XSHG    1.0
                    603727.XSHG    8.0
                    603798.XSHG    7.0
                    603800.XSHG   17.0
        '''
        df = df.reset_index()
        df = df.melt(id_vars = df.columns[0])
        df = df.set_index([ df.columns[0], df.columns[1]]).sort_index()
        df.index.names = ['date' ,      'object'  ]
        return df

    def sample_datas(self, datas, sample_n=None):
        '''把数据按类均等分'''
        if sample_n == None:
            sample_n = np.min(datas['y'].value_counts())
        datas = datas.groupby('y', group_keys=False).apply(lambda x: x.sample(sample_n))
        return datas

    def m_svd(self, df):
        U, S, V = np.linalg.svd(np.corrcoef(df.T))  # [:, dd:]
        return np.dot(df, U) / np.sqrt(S)

    def svd_datas(self, datas: pd.DataFrame, svd_n):
        '''奇异值分解'''
        datas.to_csv('datas.csv')
        datas_li = []
        dt = []
        for i in range(datas.shape[0]-svd_n+1):
            df = datas.iloc[i:i+svd_n]
            try:
                # print(self.m_svd(df).ravel())
                datas_li.append(self.m_svd(df).ravel())
                dt.append(datas.index[i+svd_n-1])
            except:
                df.to_csv('df.csv')
                print('wrong....', i)
                # exit()
        df = pd.DataFrame(datas_li)    
        df.columns = [f'column{i}' for i in range(df.shape[1])]
        print(df.shape, len(datas.index[svd_n-1:]))
        # df['datetime'] = datas.index[svd_n-1:] 
        df['datetime'] = dt
        df.set_index('datetime', inplace=True)   
        return df

    def seperate_df_class(self, df: pd.DataFrame):
        '''把数据分离开，用于板块数据处理'''
        if 'datetime' not in df.columns.to_list(): df.reset_index(inplace=True)
        df['sep'] = np.where(df['datetime']<df['datetime'].shift(1), 1, 0)
        df.set_index('datetime', inplace=True)
        index_li = [0] + df[df['sep']==1].index.to_list()
        df_res_li = []
        for i in range(len(index_li)):
            df_res_li.append(df.iloc[index_li[i]:index_li[i+1]] if i != (len(index_li)-1) else df.iloc[index_li[i]:])
        return df_res_li


class ComposeNkLine:
    '''合成k线'''
    def __init__(self, load_pa=f'{pa_prefix}/datas/data_1min', startdate=datetime(2016, 1, 1), enddate=datetime(2022, 12, 14), rq_datas=0):  # startdate=datetime(2016, 1, 1), enddate=datetime(2020, 11, 10)
        self.datas_pa = load_pa
        self.symbol_li = os.listdir(self.datas_pa)
        self.maincon = self.get_maincon()
        self.win_n_li = [5, 15, 30, 60]
        self.startdate=startdate
        self.enddate=enddate
        self.rq_datas = rq_datas

    def get_maincon(self):
        df = pd.read_csv(f'{pa_prefix}/datas/maincon.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_nk_line(self, contract, win_n=5, pa='data_5m'):
        '''跑回测获取n分钟k线'''
        symbol = get_sy(contract)
        df_maincon = self.maincon[self.maincon['contract']==contract]
        start = (df_maincon['date'].iloc[0]-timedelta(days=20)).to_pydatetime()
        end = (df_maincon['date'].iloc[-1]+timedelta(days=2)).to_pydatetime()
        if start >= self.enddate:
            return 
        else:
            if end >= self.enddate:
                end = self.enddate
            engine = BacktestingEngine()
            engine.set_parameters(
                vt_symbol=f"{contract}.LOCAL",
                interval="1m",
                start=start,
                end=end,
                rate=0,
                slippage=1,
                size=10,
                pricetick=1,
                capital=1_000_000
            )
            engine.add_strategy(NkLine, {'win_n': win_n})

            engine.load_data()
            engine.run_backtesting()
            engine.calculate_result()
            engine.calculate_statistics()
            li_res = engine.strategy.li_res
            df = pd.DataFrame(li_res)
            df.columns = engine.strategy.col
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            save_pa = f'{pa_prefix}/datas/{pa}/{symbol}/'
            makedir(save_pa)
            df.to_csv(f'{save_pa}{contract}.csv', index=False)        

            del engine

    def transfrom_datas(self, contract_li, symbol='rb'):
        '''获取和转换主力合约'''
        load_pa = f'{pa_prefix}/datas/data_1min/{symbol}/'
        save_pa = f'{pa_prefix}/datas/data_1m/{symbol}/'
        makedir(save_pa)
        
        for contract in contract_li:
            try:
                df = pd.read_csv(load_pa+contract+'.csv')
            except:
                print('have no contract:', contract)
                return 
            if self.rq_datas:
                df.rename(columns={'total_turnover': 'turnover'}, inplace=True)
            else:
                # df.columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice', 'LowerLimitPrice', 'datetime', 'Avg_price']
                df.columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'datetime', 'Avg_price']
                # df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'UpperLimitPrice', 'LowerLimitPrice']]
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['datetime'] = df['datetime'].apply(lambda x: x-timedelta(minutes=1))
            # df['datetime'] = df['datetime'].apply(self.change_datetime_sc)
            df.sort_values('datetime', ascending=True, inplace=True)
            df.to_csv(save_pa+contract+'.csv', index=False)
        print('transfrom datas done.')
    
    def change_datetime_sc(self, x):
        '''转换双璨数据'''
        if x.time() == dttime(13, 29):
            x = datetime(x.year, x.month, x.day, 11, 29)
        return x

    def save_datas(self, symbol, contract_li):
        '''将csv数据保存到数据库里'''
        main_pa = f'{pa_prefix}/datas/data_1m/{symbol}/'
        for contract in contract_li: # [ind:]
            # try:
            bars = []
            data_df = pd.read_csv(main_pa + contract + '.csv')
            data_df.dropna(inplace=True)
            data_df['datetime'] = pd.to_datetime(data_df['datetime'])
            data_df['datetime'] = data_df['datetime'].apply(lambda x: x-timedelta(hours=8))
            data_list = data_df.to_dict('records')
            for item in data_list:
                dt = datetime.fromtimestamp(item['datetime'].timestamp())
                # bar_time = dt.time()
                # is_time = time(21, 0) <= bar_time or bar_time < time(2, 30) or \
                #         time(9, 0) <= bar_time < time(10, 15) or \
                #         time(10, 30) <= bar_time < time(11, 30) or \
                #         time(13, 0) <= bar_time < time(15, 0)
                # if is_time:
                bar = BarData(
                    symbol=contract,
                    exchange=Exchange.LOCAL,
                    datetime=dt,  # datetime.fromtimestamp(item['datetime'].timestamp()),
                    interval=Interval.MINUTE,
                    open_price=float(item['open']),
                    high_price=float(item['high']),
                    low_price=float(item['low']),
                    close_price=float(item['close']),
                    volume=float(item['volume']),
                    turnover=float(item['turnover']),
                    gateway_name="DB",
                )
                bars.append(bar)
            database_manager = get_database()
            database_manager.save_bar_data(bars)
            print(symbol, contract, 'sql done.')
    
    def get_contract_li(self, symbol):
        '''获取主力合约'''
        df_maincon = self.maincon[self.maincon['symbol']==symbol.upper()]   # 获取当前品种所有合约
        df_maincon = df_maincon[(df_maincon['date'] >= self.startdate) & (df_maincon['date'] <= self.enddate)]    # 获取对应时间段的合约
        contract_li = df_maincon['contract'].unique()  
        return contract_li

    def get_symbol_datas(self, symbol='rb'):
        '''
        获取品种对应合约
        读取品种主力合约
        对主力合约进行格式转换
        保存主力合约成csv和入库
        获取合约对应的win_n_li k线, 保存成csv文件
        '''
        contract_li = self.get_contract_li(symbol)
        self.transfrom_datas(contract_li, symbol)   # 对主力合约进行格式转换，保存主力合约成csv

        self.save_datas(symbol, contract_li)    # 主力合约入库
        self.get_all_contract_k_line(contract_li)
    
    def get_all_contract_k_line(self, contract_li):
        for win_n in self.win_n_li:     # 获取合约对应的win_n_li k线, 保存成csv文件
            for contract in contract_li:
                self.get_nk_line(contract, win_n=win_n, pa=f'data_{win_n}m')
    
    def get_all_symbol_datas(self):
        '''全品种'''
        sy_li = self.symbol_li[self.symbol_li.index('PP'):]
        for symbol in self.symbol_li:
            print('start: ', symbol)
            try:
                self.get_symbol_datas(symbol)
            except:
                print(symbol, 'got wrong.')
            print('end: ', symbol)


class DistributeAnalyze:
    '''数据分析'''
    def __init__(self):
        self.interval_li = [1, 5, 15, 30, 60]
        self.bins_li = [100, 100, 70, 70, 50]
        self.win_n = np.array(self.interval_li)     # 前n根k线
        self.pred_h = [1, 2, 3, 6]  # n小时的收益率窗口
        self.quantile_n = [i for i in range(2, 5)]
        self.maincon = get_maincon()
        self.count = 0

    def plot_quantile(self, df: pd.DataFrame, save_pa, save_name, interval=1, quantile_n=5, groupby='quantile', columns='sharp_ratio'):
        '''画分位数对应的分布图'''
        bins = self.bins_li[self.interval_li.index(interval)] // 2
        # bins = 5
        fig, axes = plt.subplots(1, quantile_n, figsize=(20, 15))
        plt.suptitle(save_name,fontsize=20)
        step = 100 // quantile_n
        co = 1
        for ax, data in zip(axes.flatten(), df.groupby(groupby)): 
            data = list(data)[1][columns]
            data_name = data.name
            ax.hist(data, bins, density=True)
            ax.set(xlabel=f'{data_name}_{step*co}', ylabel='')
            co += 1
            plt.savefig(f'{save_pa}{save_name}_{columns}_{quantile_n}.png')
        plt.close()

    def _caculate_dir(self, df, n, columns, res_dir, quantile='quantile'):
        '''计算偏度存到字典里'''
        step = 100 // n
        res_dir['plot_n'].append(n)
        df_de1 = df[df[quantile]==0]
        df_de11 = df_de1[df_de1[columns]<0]
        df_de12 = df_de1[df_de1[columns]>0]
        rate1 = df_de11.shape[0] / df_de1.shape[0]
        deviation1 = '左偏' if rate1 > 0.5 else '右偏'
        df_de2 = df[df[quantile]==step*(n-1)]
        df_de21 = df_de2[df_de2[columns]>0]
        df_de22 = df_de2[df_de2[columns]<0]
        rate2 = df_de21.shape[0] / df_de2.shape[0]
        deviation2 = '右偏' if rate2 > 0.5 else '左偏'
        res_dir['deviation1'].append(deviation1), res_dir['deviation2'].append(deviation2)
        res_dir['deviation1_ratio'].append(rate1), res_dir['deviation2_ratio'].append(rate2)
        res_dir['deviation1_mean_left'].append(df_de11[columns].mean()), res_dir['deviation2_mean_right'].append(df_de21[columns].mean())
        res_dir['deviation1_mean_right'].append(df_de12[columns].mean()), res_dir['deviation2_mean_left'].append(df_de22[columns].mean())
        return res_dir
        
    def get_quantile(self, df: pd.DataFrame, save_pa, save_name, interval=1, quantile_n=[5], quantile_col='high_low_pct', columns='sharp_ratio', need_sharp_ratio=1, res_dir={}):
        '''获取涨跌幅分位数进行夏普比率分类'''
        is_first = 1
        for n in quantile_n:
            step = 100 // n
            df['quantile'] = np.nan
            q = df[quantile_col]
            for i in range(0, step*n, step):
                df['quantile'] = np.where((q>=q.quantile(i/100)) & (q<=q.quantile((i+step)/100)), i, df['quantile'])
            df.dropna(inplace=True)

            if is_first:
                is_first = 0
            else:
                res_dir['contract'].append(res_dir['contract'][-1]), res_dir['interval'].append(res_dir['interval'][-1])
                res_dir['x_lable'].append(res_dir['x_lable'][-1]), res_dir['x_interval'].append(res_dir['x_interval'][-1])
                res_dir['y_lable'].append(res_dir['y_lable'][-1]), res_dir['y_interval'].append(res_dir['y_interval'][-1])

            res_dir = self._caculate_dir(df, n, columns, res_dir, quantile='quantile')
            # if n==4:
            #     df.to_csv('df_test.csv')
            #     print('store--------')
            self.plot_quantile(df, save_pa, f'{save_name}_{quantile_col}', interval, n, columns=columns)
        return res_dir

    def _get_quantile_all(self, high_low_li, columns_all, df, interval, save_pa, s0, need_sharp_ratio, res_dir, contract):
        '''计算不同涨跌幅下的不同时间窗口的收益率和夏普比率分布情况'''
        for quantile_col in high_low_li:
            x_interval = quantile_col.split('_')[-1]       
            for columns in columns_all:
                if need_sharp_ratio or ('sharp_ratio' not in columns):
                    y_interval = columns.split('_')[-1]
                    res_dir['contract'].append(contract), res_dir['interval'].append(interval)
                    res_dir['x_lable'].append(quantile_col.replace('_'+x_interval, '')), res_dir['x_interval'].append(x_interval), 
                    res_dir['y_lable'].append(columns.replace('_'+y_interval, '')), res_dir['y_interval'].append(y_interval)
                    res_dir = self.get_quantile(df, save_pa, s0, interval, self.quantile_n, quantile_col, columns, need_sharp_ratio, res_dir)
        return res_dir
                
    def high_low_win_n(self, df: pd.DataFrame, win_n, interval):
        '''获取前n根k线的最高价和最低价'''
        col_li = []
        for i in win_n:
            df['high_n'] = df['high'].rolling(i).max()
            df['low_n'] = df['low'].rolling(i).min()
            df['close_n'] = df['close'].shift(i)
            str_i = f'high_low_pct_{str(i*interval)}m'
            df[str_i] = np.sign((df['close']-df['close_n']) / df['close_n']) * (df['high_n'] - df['low_n']) / df['close_n']
            col_li.append(str_i)
        return df, col_li
    
    def pred_nh_return_rate_sharp_ratio(self, df: pd.DataFrame, pred_h, interval):
        '''后n小时的收益率'''
        return_rate_li, sharp_ratio_li = [], []

        for i in pred_h:
            shift_n = (60 // interval) * i
            str_1 = f'return_rate_{str(i)}h'
            str_2 = f'sharp_ratio_{str(i)}h'
            return_rate_li.append(str_1)
            sharp_ratio_li.append(str_2)
            df[str_1] = df['close'].pct_change(shift_n).shift(-shift_n)
            df[str_2] = df['return_rate'].rolling(shift_n).apply(sharp_ratio).shift(-shift_n)
        return df, return_rate_li, sharp_ratio_li

    def m_hist(self, df, columns, save_name, save_pa, bins, need_sharp_ratio):
        if not need_sharp_ratio:
            save_name = [save_name[0]]
            columns = [columns[0]]
        for s, column in zip(save_name, columns):
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            plt.suptitle(s,fontsize=20)
            datas = [df[col] for col in column]
            # datas = [df[f'return_rate_{str(i)}h'] for i in self.pred_h]
            for ax, data in zip(axes.flatten(), datas):
                ax.hist(data, bins, density=True)
                ax.set(xlabel=data.name, ylabel='')
                plt.savefig(f'{save_pa}{s}.png')
            plt.close()
        
    def plot_hist(self, symbol='rb', interval=30):
        '''画一个品种某分钟k线的分布图'''
        def m_plot(df, s, save_pa):
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            plt.suptitle(s,fontsize=20)
            datas = [df[['datetime', 'close']], df['return_rate'], df[f'high_low_pct_{str(interval)}m']]
            df_li.append(df)
            if need_sharp_ratio:
                datas.append(df[f'sharp_ratio_1h'])
            for ax, data in zip(axes.flatten(), datas):
                if pd.DataFrame(data).shape[1] == 2:
                    ax.plot(data.iloc[:, 0],data.iloc[:, 1])
                    ax.set(xlabel=data.columns[0], ylabel=data.columns[1])
                    ax.set_xticks([0, len(data)/2, len(data)-1])
                else:
                    ax.hist(data, bins, density=True)
                    ax.set(xlabel=data.name, ylabel='')
                    # ax.set_xticks(x_label)
                    plt.savefig(f'{save_pa}{s}.png')
            plt.close()
        
        
        load_pa = f'{pa_prefix}/datas/data_{interval}m/{symbol}/'
        # save_pa = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/'
        try:
            li = os.listdir(load_pa)
        except:
            print(load_pa, 'is not exist')
            return 
        if len(li) == 0:
            return
        # makedir(save_pa)
        shift_n = 60 // interval
        if interval == 30 or interval == 60:
            need_sharp_ratio = 0
        else:
            need_sharp_ratio = 1

        bins = self.bins_li[self.interval_li.index(interval)]
        df_li = []
        win_n = list(self.win_n // interval)
        win_n = win_n[win_n.index(1):]
        
        for pa in li:
            contract = pa.split('.')[0]
            save_pa = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/{contract}/'
            makedir(save_pa)
            s0 = f'{contract}_{interval}m'
            df_maincon = self.maincon[self.maincon['contract']==contract]
            start_date, end_date = df_maincon['date'].iloc[0], df_maincon['date'].iloc[-1]
            df = pd.read_csv(load_pa+pa)
            df['return_rate'] = df['close'].pct_change()
            df, return_rate_li, sharp_ratio_li = self.pred_nh_return_rate_sharp_ratio(df, self.pred_h, interval)
            columns_all = return_rate_li + sharp_ratio_li
            # df['high_low_pct'] = np.sign(df['return_rate']) * (df['high'] - df['low']) / df['open']
            df, high_low_li = self.high_low_win_n(df, win_n, interval)
            df.dropna(inplace=True)     # 前n根k线涨跌幅对应的夏普比率分布图

            self._get_quantile_all(high_low_li, columns_all, df, interval, save_pa, s0, need_sharp_ratio=need_sharp_ratio)

            df['dt'] = pd.to_datetime(df['datetime'])
            df = df[(df['dt'].dt.date >= start_date) & (df['dt'].dt.date <= end_date)]
            sharp = sharp_ratio(df['return_rate'])
            
            s1 = f'{s0}_{sharp}'
            # self.get_quantile(df, save_pa, s0, interval, quantile_n=quantile_n)
            m_plot(df, s1, save_pa)
            self.m_hist(df, [return_rate_li, sharp_ratio_li], [f'{s0}_return_rate', f'{s0}_sharp_ratio'], save_pa, bins, need_sharp_ratio)
            print(contract, interval)

        df_concat = pd.concat(df_li)
        sharp = sharp_ratio(df_concat['return_rate'])
        s0 = f'{symbol}_{interval}m'
        s1 = f'{s0}_{sharp}'
        save_pa_all = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/{symbol}_all/'
        makedir(save_pa_all)
        m_plot(df_concat, s1, save_pa_all)
        self.m_hist(df_concat, [return_rate_li, sharp_ratio_li], [f'{s0}_return_rate', f'{s0}_sharp_ratio'], save_pa_all, bins, need_sharp_ratio)
        self._get_quantile_all(high_low_li, columns_all, df_concat, interval, save_pa_all, s0, need_sharp_ratio=need_sharp_ratio)

    def get_high_low_pct_return_rate_sharp_ratio(self, df: pd.DataFrame, interval, need_sharp_ratio):
        '''获取涨跌幅，收益率和夏普比率, 长周期预测短周期'''
        def m_func(df, i, j, interval, m='m'):
            str_s = ''
            if need_sharp_ratio and m == 'h':
                str_s = f'sharp_ratio_{j*interval//60}h'
                df[str_s] = df['return_rate'].rolling(j).apply(sharp_ratio).shift(-j)
            df['high_n'] = df['high'].rolling(i).max()
            df['low_n'] = df['low'].rolling(i).min()
            df['close_n'] = df['close'].shift(i)
            str_i = f'high_low_pct_{str(i*interval)}m' if m == 'm' else f'high_low_pct_{i*interval//60}h'
            df[str_i] = np.sign((df['close']-df['close_n']) / df['close_n']) * (df['high_n'] - df['low_n']) / df['close_n']
            str_j = f'return_rate_{j*interval}m' if m == 'm' else f'return_rate_{j*interval//60}h'
            df[str_j] = df['close'].pct_change(j).shift(-j)
            return df, str_i, str_j, str_s

        high_low_n = [5, 10]    # 含义：周期，小时
        return_rate_n = [1, 2]
        high_low_li_m, high_low_li_h, return_rate_li_m, return_rate_li_h, sharp_ratio_li = [], [], [], [], []
        for i, j in zip(high_low_n, return_rate_n):
            df, str_im, str_jm, _ = m_func(df, i, j, interval, 'm') # 按周期
            df, str_ih, str_jh, str_s = m_func(df, 60//interval*i, 60//interval*j, interval, 'h') # 按小时
            high_low_li_m.append(str_im), high_low_li_h.append(str_ih)
            return_rate_li_m.append(str_jm), return_rate_li_h.append(str_jh)
            if len(str_s) != 0:
                sharp_ratio_li.append(str_s)
        return df, high_low_li_m, high_low_li_h, return_rate_li_m, return_rate_li_h, sharp_ratio_li

    def plot_hist1(self, symbol='rb', interval=30):
        '''画一个品种某分钟k线的分布图'''
        load_pa = f'{pa_prefix}/datas/data_{interval}m/{symbol}/'
        try:
            li = os.listdir(load_pa)
        except:
            print(load_pa, 'is not exist')
            return 
        if len(li) == 0:
            return
        shift_n = 60 // interval
        if interval == 30 or interval == 60:
            need_sharp_ratio = 0
        else:
            need_sharp_ratio = 1
        df_li = []
        win_n = list(self.win_n // interval)
        win_n = win_n[win_n.index(1):]
        
        for pa in li:
            contract = pa.split('.')[0]
            save_pa = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/{contract}/'
            makedir(save_pa)
            s0 = f'{contract}_{interval}m'
            df_maincon = self.maincon[self.maincon['contract']==contract]
            start_date, end_date = df_maincon['date'].iloc[0], df_maincon['date'].iloc[-1]
            df = pd.read_csv(load_pa+pa)
            df['return_rate'] = df['close'].pct_change()
            df, high_low_li_m, high_low_li_h, return_rate_li_m, return_rate_li_h, sharp_ratio_li = self.get_high_low_pct_return_rate_sharp_ratio(
                df, interval, need_sharp_ratio)
            df.dropna(inplace=True)     # 前n根k线涨跌幅对应的夏普比率分布图

            self._get_quantile_all(high_low_li_m, return_rate_li_m, df, interval, save_pa, s0, need_sharp_ratio=need_sharp_ratio)
            self._get_quantile_all(high_low_li_h, return_rate_li_h+sharp_ratio_li, df, interval, save_pa, s0, need_sharp_ratio=need_sharp_ratio)

            df['dt'] = pd.to_datetime(df['datetime'])
            df = df[(df['dt'].dt.date >= start_date) & (df['dt'].dt.date <= end_date)]
            df_li.append(df)
            print(contract, interval)

        df_concat = pd.concat(df_li)
        s0 = f'{symbol}_{interval}m'
        save_pa_all = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/{symbol}_all/'
        makedir(save_pa_all)
        self._get_quantile_all(high_low_li_m, return_rate_li_m, df_concat, interval, save_pa_all, s0, need_sharp_ratio=need_sharp_ratio)
        self._get_quantile_all(high_low_li_h, return_rate_li_h+sharp_ratio_li, df_concat, interval, save_pa_all, s0, need_sharp_ratio=need_sharp_ratio)

    def plot_hist_adj(self, symbol='rb', interval=30, res_dir={}):
        '''画一个品种某分钟k线的分布图'''
        # res_dir = {'contract': [], 'interval': [], 'x_lable': [], 'x_interval': [], 'y_lable': [], 'y_interval': [],
        #             'plot_n': [], 'deviation1': [], 'deviation2': [], 'deviation1_ratio': [], 'deviation2_ratio': [],
        #             'deviation1_mean': [], 'deviation2_mean': []}
        def m_plot(df, s, save_pa):
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            plt.suptitle(s,fontsize=20)
            datas = [df[['datetime', 'close']], df['return_rate'], df[f'high_low_pct_{str(interval)}m']]
            if need_sharp_ratio:
                datas.append(df[f'sharp_ratio_1h'])
            for ax, data in zip(axes.flatten(), datas):
                if pd.DataFrame(data).shape[1] == 2:
                    ax.plot(data.iloc[:, 0],data.iloc[:, 1])
                    ax.set(xlabel=data.columns[0], ylabel=data.columns[1])
                    ax.set_xticks([0, len(data)/2, len(data)-1])
                else:
                    ax.hist(data, bins, density=True)
                    ax.set(xlabel=data.name, ylabel='')
                    # ax.set_xticks(x_label)
                    plt.savefig(f'{save_pa}{s}.png')
            plt.close()
        
        
        load_pa = f'{pa_prefix}/datas/data_{interval}m/{symbol}/'
        # save_pa = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/'
        try:
            li = os.listdir(load_pa)
        except:
            print(load_pa, 'is not exist')
            return 
        if len(li) == 0:
            return
        # makedir(save_pa)
        shift_n = 60 // interval
        if interval == 30 or interval == 60:
            need_sharp_ratio = 0
        else:
            need_sharp_ratio = 1

        bins = self.bins_li[self.interval_li.index(interval)]
        df_li = []
        win_n = list(self.win_n // interval)
        win_n = win_n[win_n.index(1):]
        
        for pa in li:
            contract = pa.split('.')[0]
            
            save_pa = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/{contract}/'
            makedir(save_pa)
            s0 = f'{contract}_{interval}m'
            df_maincon = self.maincon[self.maincon['contract']==contract]
            start_date, end_date = df_maincon['date'].iloc[0], df_maincon['date'].iloc[-1]
            df = pd.read_csv(load_pa+pa)
            df['return_rate'] = df['close'].pct_change()
            df, return_rate_li, sharp_ratio_li = self.pred_nh_return_rate_sharp_ratio(df, self.pred_h, interval)
            df, high_low_li = self.high_low_win_n(df, win_n, interval)
            df, high_low_li_m, high_low_li_h, return_rate_li_m, return_rate_li_h, sharp_ratio_li1 = self.get_high_low_pct_return_rate_sharp_ratio(
                df, interval, need_sharp_ratio)
            df.dropna(inplace=True)     # 前n根k线涨跌幅对应的夏普比率分布图

            res_dir = self._get_quantile_all(high_low_li_m, return_rate_li_m, df, interval, save_pa, s0, need_sharp_ratio, res_dir, contract)
            res_dir = self._get_quantile_all(high_low_li_h, return_rate_li_h+sharp_ratio_li1, df, interval, save_pa, s0, need_sharp_ratio, res_dir, contract)

            df['dt'] = pd.to_datetime(df['datetime'])
            df = df[(df['dt'].dt.date >= start_date) & (df['dt'].dt.date <= end_date)]
            sharp = sharp_ratio(df['return_rate'])
            
            s1 = f'{s0}_{sharp}'
            df_li.append(df)
            m_plot(df, s1, save_pa)
            self.m_hist(df, [return_rate_li, sharp_ratio_li], [f'{s0}_return_rate', f'{s0}_sharp_ratio'], save_pa, bins, need_sharp_ratio)
            print(contract, interval)

        df_concat = pd.concat(df_li)
        sharp = sharp_ratio(df_concat['return_rate'])
        s0 = f'{symbol}_{interval}m'
        s1 = f'{s0}_{sharp}'
        save_pa_all = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/{symbol}_all/'
        makedir(save_pa_all)
        m_plot(df_concat, s1, save_pa_all)
        sy = symbol+'_all'
        self.m_hist(df_concat, [return_rate_li, sharp_ratio_li], [f'{s0}_return_rate', f'{s0}_sharp_ratio'], save_pa_all, bins, need_sharp_ratio)
        res_dir = self._get_quantile_all(high_low_li_m, return_rate_li_m, df_concat, interval, save_pa_all, s0, need_sharp_ratio, res_dir, sy)
        res_dir = self._get_quantile_all(high_low_li_h, return_rate_li_h+sharp_ratio_li1, df_concat, interval, save_pa_all, s0, need_sharp_ratio, res_dir, sy)
        # self.save_res_dir(symbol, res_dir)
        return res_dir

    def save_res_dir(self, symbol, res_dir):
        '''保存结果'''
        pa = f'{pa_prefix}/datas/data_plot/datas_analyze/{symbol}/res/'
        makedir(pa)
        df = pd.DataFrame(res_dir)
        df.to_csv(f'{pa}{symbol}_res.csv', encoding="utf_8_sig", index=False)

    def plot_symbol_hist(self, symbol='rb'):
        '''画一个品种的分布图'''
        res_dir = {'contract': [], 'interval': [], 'x_lable': [], 'x_interval': [], 'y_lable': [], 'y_interval': [],
                    'plot_n': [], 'deviation1': [], 'deviation2': [], 'deviation1_ratio': [], 'deviation2_ratio': [],
                    'deviation1_mean_left': [], 'deviation1_mean_right': [], 'deviation2_mean_left': [], 'deviation2_mean_right': []}
        for interval in self.interval_li:
            res_dir = self.plot_hist_adj(symbol, interval, res_dir)
        self.save_res_dir(symbol, res_dir)
        print(symbol, 'is done.')

    def plot_all_hist(self, max_workers=5):
        '''画所有品种的分布图'''
        symbol_li = os.listdir('{pa_prefix}/datas/data_1m/')
        # ind = symbol_li.index('RS')
        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # max_workers=10
            executor.map(self.plot_symbol_hist, symbol_li)


class DataProcessML(BaseDataProcess):
    '''机器学习数据处理'''
    def __init__(self, need_test_set=1, time_series=0) -> None:
        super().__init__()
        self.datas_pa = f'{pa_prefix}/datas/data_index/'
        self.save_datas_pa = f'{pa_prefix}/datas/data_set/'
        self.time_series = time_series
        self.need_test_set = need_test_set
        self.train_val_test_date()
        self.train_set, self.val_set, self.test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.factor_normalize = pd.DataFrame()
        self.syinfo = SymbolsInfo()
        self.need_log = 0

    def train_val_test_date(self):
        if self.need_test_set:
            # self.train_date, self.val_date, self.test_date = datetime(2016, 1, 1), datetime(2020, 5, 1), datetime(2021, 5, 1)
            self.train_date, self.val_date, self.test_date = datetime(2016, 1, 1), datetime(2019, 5, 1), datetime(2020, 5, 1)
        else:
            self.train_date, self.val_date = datetime(2016, 1, 1), datetime(2019, 11, 1)

    def load_datas(self, symbol, pa):
        '''读取数据'''
        df = pd.read_csv(f'{self.datas_pa}{symbol}/{pa}.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
        
    def process_y(self, df: pd.Series, y_thread):
        '''
        处理y值:
        将0.2和0.8分位数分别标0和2, 其余标1
        '''
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
            if y_method == 2:
                y_std = df.shift(-1).rolling(n).std().shift(-n)
                y = np.ones(len(y_std))
                y_mean = df.shift(-1).rolling(n).mean().shift(-n) if need_mean else 0
                y = np.where(df>abs(y_mean+k*y_std), 2, y)
                y = np.where(df<-abs(y_mean+k*y_std), 0, y)
        
        elif y_thread == 0:
            y = df.apply(np.sign)
            
        else:
            df_train = df[df.index<self.val_date]
            value_thread = df_train.abs().quantile(y_thread)
            y = df.apply(classify)

        return y
        
    def _del_y_0(self, df, y_thread):
        if y_thread is 0:
            df_i = df[df['y']!=0]
            df_i['y'] = df_i['y'].apply(lambda x: 0 if x<0 else x)
            return df_i
        else:
            return df

    # def process_factors(self, df: pd.DataFrame, y_thread, method, need_svd=False):
    #     '''
    #     因子数据处理:
    #     去极值
    #     标准化
    #     '''
    #     def select_columns(x):
    #         '''获取不需要去极值的列'''
    #         if len(x.unique()) > 3:
    #             return 0
    #         else:
    #             return 1
        
    #     sc = df.apply(select_columns)
    #     sc = sc[sc==1].index.to_list()
        
    #     factor_two_kind = df.loc[:,df.columns.isin(sc)]
    #     sc.append('y')
    #     factor = df.loc[:,~df.columns.isin(sc)]

    #     factor_two_kind = factor_two_kind.apply(np.sign)
    #     factor_two_kind['y'] = self.process_y(df['y'], y_thread)

    #     # if method == 0:
    #     factor_train = factor[factor.index<self.val_date]
    #     # else:
    #         # factor_train = factor[factor.index<self.test_date]

    #     try:
    #         dm_series = self.cal_dm_and_d1(factor_train)
    #         factor = self.mad(factor, dm_series)
    #     except:
    #         return pd.DataFrame(), 0, 0

    #     # 平稳时间序列
    #     # adf_res = factor.apply(self.adf_test)
    #     # adf_res = pd.DataFrame(adf_res)
    #     # adf_res = adf_res.reset_index()
    #     # adf_res.columns = ['ind', 'is_smooth']
    #     # adf_res = adf_res[adf_res['is_smooth']==True]
    #     # print(len(adf_res))

    #     # adf_res.to_csv('adf_res.csv')

    #     # 标准化
    #     dic_normal = self.cal_stats(factor[factor.index<self.val_date])
    #     factor_normal = self.znormal(factor, dic_normal)

    #     # 奇异值分解
    #     if need_svd:    
    #         factor_normal = self.svd_datas(factor_normal, 12)
        
    #     # 添加shift指标
    #     # factor_normal = mff.get_shift(factor_normal, self.shift_n)  
    #     # factor_normal = factor_normal.iloc[self.shift_n:]
    #     factor_normal = pd.merge(factor_normal, factor_two_kind, left_index=True, right_index=True)

    #     factor_normal.dropna(inplace=True)
    #     factor_normal = self._del_y_0(factor_normal, y_thread)
    #     # 添加板块标签
    #     # factor_normal['sector'] = self.symbols_classify[self.symbols_classify['symbol']==symbol]['sector'].iloc[0]

    #     # 保存数据
    #     # if symbol != None:
    #     #     factor_normal['symbol'] = symbol
    #     #     factor_normal.to_csv(f'{self.save_datas_pa}{symbol}.csv')     # 保存到csv

    #     # 划分训练集和测试集
    #     if method == 0:
    #         if self.need_test_set:
    #             train_set = factor_normal[factor_normal.index<self.val_date]
    #             val_test_set = factor_normal[factor_normal.index>=self.val_date]
    #             val_set = val_test_set[val_test_set.index<self.test_date]
    #             test_set = val_test_set[val_test_set.index>=self.test_date]
    #             df_dic = {'train_datas': train_set, 'val_datas': val_set, 'test_datas': test_set, 'normalize_datas': factor_normal}

    #             if self.time_series:
    #                 for i, df in df_dic.items():
    #                     df_dic[i] = self.time_series_process(df)
                
    #             self.print_sample_num(df_dic)
    #         else:
    #             train_set = factor_normal[factor_normal.index<self.val_date] 
    #             val_set = factor_normal[factor_normal.index>=self.val_date]
    #             df_dic = {'train_datas': train_set, 'val_datas': val_set, 'normalize_datas': factor_normal}
    #             self.print_sample_num(df_dic)
        
    #     elif method == 1:
    #         train_set = factor_normal[factor_normal.index<self.test_date] 
    #         test_set = factor_normal[factor_normal.index>=self.test_date]
    #         df_dic = {'train_datas': train_set, 'test_datas': test_set, 'normalize_datas': factor_normal}
    #         self.print_sample_num(df_dic)

    #     return df_dic

    def process_factors(self, df: pd.DataFrame, y_thread, method, need_svd=False):
        '''
        因子数据处理:
        加log
        去极值
        标准化
        '''
        def select_columns(x):
            '''获取不需要去极值的列'''
            if len(x.unique()) > 3:
                return 0
            else:
                return 1
        
        sc = df.apply(select_columns)
        sc = sc[sc==1].index.to_list()
        
        factor_two_kind = df.loc[:,df.columns.isin(sc)]
        sc.append('y')
        factor = df.loc[:,~df.columns.isin(sc)]

        # 加log
        # if self.need_log: factor = factor.apply(lambda x: np.sign(x)*np.log(1+np.abs(x)))

        factor_two_kind = factor_two_kind.apply(np.sign)
        factor_two_kind['y'] = self.process_y(df['y'], y_thread)

        # if method == 0:
        factor_train = factor[factor.index<self.val_date]
        # else:
            # factor_train = factor[factor.index<self.test_date]

        try:
            dm_series = self.cal_dm_and_d1(factor_train)
            factor = self.mad(factor, dm_series)
        except:
            return pd.DataFrame(), 0, 0

        # 平稳时间序列
        # adf_res = factor.apply(self.adf_test)
        # adf_res = pd.DataFrame(adf_res)
        # adf_res = adf_res.reset_index()
        # adf_res.columns = ['ind', 'is_smooth']
        # adf_res = adf_res[adf_res['is_smooth']==True]
        # print(len(adf_res))

        # adf_res.to_csv('adf_res.csv')

        # 标准化
        dic_normal = self.cal_stats(factor[factor.index<self.val_date])
        factor_normal = self.znormal(factor, dic_normal)

        # 奇异值分解
        if need_svd:    
            factor_normal = self.svd_datas(factor_normal, 12)
        
        # 添加shift指标
        # factor_normal = mff.get_shift(factor_normal, self.shift_n)  
        # factor_normal = factor_normal.iloc[self.shift_n:]
        factor_normal = pd.merge(factor_normal, factor_two_kind, left_index=True, right_index=True)

        factor_normal.dropna(inplace=True)
        factor_normal = self._del_y_0(factor_normal, y_thread)
        # 添加板块标签
        # factor_normal['sector'] = self.symbols_classify[self.symbols_classify['symbol']==symbol]['sector'].iloc[0]

        # 保存数据
        # if symbol != None:
        #     factor_normal['symbol'] = symbol
        #     factor_normal.to_csv(f'{self.save_datas_pa}{symbol}.csv')     # 保存到csv

        # 划分训练集和测试集
        if method == 0:
            if self.need_test_set:
                train_set = factor_normal[factor_normal.index<self.val_date]
                val_test_set = factor_normal[factor_normal.index>=self.val_date]
                val_set = val_test_set[val_test_set.index<self.test_date]
                test_set = val_test_set[val_test_set.index>=self.test_date]
                df_dic = {'train_datas': train_set, 'val_datas': val_set, 'test_datas': test_set, 'normalize_datas': factor_normal}

                if self.time_series:
                    for i, df in df_dic.items():
                        df_dic[i] = self.time_series_process(df)
                
                self.print_sample_num(df_dic)
            else:
                train_set = factor_normal[factor_normal.index<self.val_date] 
                val_set = factor_normal[factor_normal.index>=self.val_date]
                df_dic = {'train_datas': train_set, 'val_datas': val_set, 'normalize_datas': factor_normal}
                self.print_sample_num(df_dic)
        
        elif method == 1:
            train_set = factor_normal[factor_normal.index<self.test_date] 
            test_set = factor_normal[factor_normal.index>=self.test_date]
            df_dic = {'train_datas': train_set, 'test_datas': test_set, 'normalize_datas': factor_normal}
            self.print_sample_num(df_dic)

        return df_dic

    def process_factors_class(self, df: pd.DataFrame, y_thread, method, need_svd=False):
        '''按板块里每个品种进行数据处理'''
        df_li = self.seperate_df_class(df)
        df_dic = {'train_datas': [], 'val_datas': [], 'test_datas': [], 'normalize_datas': []}
        for df_i in df_li:
            df_i_dic = self.process_factors(df_i, y_thread, method, need_svd=need_svd)
            df_dic['train_datas'].append(df_i_dic['train_datas'])
            df_dic['val_datas'].append(df_i_dic['val_datas'])
            df_dic['test_datas'].append(df_i_dic['test_datas'])
            df_dic['normalize_datas'].append(df_i_dic['normalize_datas'])
        for key, val in df_dic.items():
            df_dic[key] = pd.concat(val)

        return df_dic
        
    def process_factors_test(self, df: pd.DataFrame, dm_series_pa, dic_normal_pa, save_pa, is_save=0):
        '''
        因子数据处理:
        去极值
        标准化
        用于MLTest
        '''

        # try:
        #     dm_series = joblib.load(dm_series_pa)    # dataframe
        #     dic_normal = joblib.load(dic_normal_pa)  # dic
        # except:
        #     df_train = df[df.index<self.val_date]
        #     dm_series = bdp.cal_dm_and_d1(df_train) # #去极值
        #     dic_normal = bdp.cal_stats(df_train)  # 标准化
        #     joblib.dump(dm_series, dm_series_pa)
        #     joblib.dump(dic_normal, dic_normal_pa)

        def select_columns(x):
            '''获取不需要去极值的列'''
            if len(x.unique()) > 3:
                return 0
            else:
                return 1
        
        sc = df.apply(select_columns)
        sc = sc[sc==1].index.to_list()
        
        factor_two_kind = df.loc[:,df.columns.isin(sc)]
        factor = df.loc[:,~df.columns.isin(sc)]

        # 加log
        if self.need_log: factor = factor.apply(lambda x: np.sign(x)*np.log(1+np.abs(x)))

        factor_two_kind = factor_two_kind.apply(np.sign)
        factor_train = factor[factor.index<self.val_date]

        # try:
        if is_save:
            dm_series = self.cal_dm_and_d1(factor_train)
            joblib.dump(dm_series, dm_series_pa)
            dic_normal = self.cal_stats(factor[factor.index<self.val_date])
            joblib.dump(dic_normal, dic_normal_pa)
        else:
            dm_series = joblib.load(dm_series_pa)    # dataframe
            dic_normal = joblib.load(dic_normal_pa)  # dic
        factor = self.mad(factor, dm_series)

        # 标准化
        factor_normal = self.znormal(factor, dic_normal)
        factor_normal = pd.merge(factor_normal, factor_two_kind, left_index=True, right_index=True)
        # factor_normal.dropna(inplace=True)
        factor_normal.to_csv(save_pa)
        # print('done.')
        # input()
        return factor_normal

    def time_series_process(self, df: pd.DataFrame):
        '''时间序列处理，将时间展开，用于做CNN或RNN'''
        res_li = []
        for i in range(len(df)-self.time_series+1):
            res_li.append(df.iloc[i:i+self.time_series].values.flatten())
        
        df_res = pd.DataFrame(res_li)
        df_res.index = df.index[self.time_series-1:]
        return df_res

    def print_sample_num(self, df_dic):
        for i in df_dic:
            print(i)
            print(df_dic[i]['y'].value_counts())
        
    def save_datas(self, symbol, suffix, index_n, df_dic):
        '''保存训练集验证集和测试集数据'''
        pa = f'{self.save_datas_pa}{symbol}/'
        makedir(pa)
        for name in df_dic:
            # print(f'{pa}{name}_{suffix}.csv', 'done......')
            df_dic[name].to_csv(f'{pa}{name}_{suffix}.csv')
        # self.train_set.to_csv(f'{pa}train_datas_{suffix}.csv')
        # self.val_set.to_csv(f'{pa}val_datas_{suffix}.csv')
        # self.test_set.to_csv(f'{pa}test_datas_{suffix}.csv')
        # self.factor_normalize.to_csv(f'{pa}normalize_datas_{suffix}.csv')
        print('save done.')

    def save_columns(self, df: pd.DataFrame, symbol, suffix):
        '''保存columns'''
        pa = f'{pa_prefix}/datas/datas_columns/{symbol}/'
        makedir(pa)
        col = pd.DataFrame(df.columns.to_list())
        col.columns = ['columns']
        pa_suffix = f'{pa}{suffix}'
        col.to_csv(f'{pa_suffix}.csv', index=False)
        fis = FactorIndexStatistics()
        fis.index_category(pa_suffix)

    def seperate_trend_shock(self, symbol, df_dic):
        '''将训练集验证集和测试集数据按趋势项和震荡项分离'''
        zigzaginfo = ZigZagInfo()
        res_dic = {}
        for key, df_i in df_dic.items():
            df_trend_res, df_shock_res = zigzaginfo.seperate_trend_shock_data(symbol, df_i.resetindex())
            res_dic.update({f'{key}_trend': df_trend_res.set_index('datetime'), f'{key}_shock': df_shock_res.set_index('datetime')})
        return res_dic

    def run_datas_process(self, symbol, pa=None, y_thread=0.5, index_n=10, method=0, need_svd=False, is_zigzag=0):
        '''执行数据处理
            1、读取数据
            2、去极值标准化
            3、保存数据
        '''
        df = self.load_datas(symbol, pa)
        if df.shape[1]-1 > index_n:
            columns = [0] + list(np.random.permutation(range(1, df.shape[1])))[:index_n]
            df = df.iloc[:, columns]
        else:
            index_n = df.shape[1]-1
        
        suffix = f'{y_thread}_{pa}'
        self.save_columns(df, symbol, suffix)   # 保存指标名称

        if symbol in self.syinfo.symbol_li:
            df_dic = self.process_factors(df, y_thread, method, need_svd=need_svd)
        else:
            df_dic = self.process_factors_class(df, y_thread, method, need_svd=need_svd)

        if is_zigzag:
            res_dic = self.seperate_trend_shock(symbol, df_dic)
            self.save_datas(symbol, suffix, index_n, res_dic)
                
        self.save_datas(symbol, suffix, index_n, df_dic)
        return index_n, suffix


def run_nkl(symbol, rq_datas=0):
    nkl = ComposeNkLine(rq_datas=rq_datas)
    nkl.get_symbol_datas(symbol)
    # nkl.get_all_symbol_datas()
    print('run_nkl is done.')

def run_nkl_all(symbol_li, rq_datas=0):
    for symbol in symbol_li:  # 合成k线
        run_nkl(symbol, rq_datas=rq_datas)
    print('run_nkl_all is done.')

def run_datas_analyze():
    da = DistributeAnalyze()
    # da.plot_hist('rb', 15)
    # da.plot_hist_adj('rb', 15)
    s = ['rb', 'j', 'm', 'p', 'ru']
    # for sy in s:
    da.plot_symbol_hist('ru')
    # da.plot_all_hist(5)
    print('run_da is done.')

def run_dp(symbol, suffix, y_thread=0.5, index_n=10, method=0, need_test_set=1, need_svd=False, time_series=0):
    '''数据处理和保存训练集验证集和测试集'''
    dp = DataProcessML(need_test_set, time_series=time_series)
    index_n, suffix = dp.run_datas_process(symbol, suffix, y_thread, index_n, method, need_svd)
    return index_n, suffix


if __name__ == '__main__':
    # run_dp()
    # symbol_li = os.listdir(f'{pa_prefix}/datas/data_1min/')
    # run_nkl_all(symbol_li, rq_datas=1)
    # run_change_maincon_to_rq()
    nkl = ComposeNkLine(rq_datas=0)
    nkl.get_all_symbol_datas()
    # nkl.get_symbol_datas('RB')
    
    # nkl.get_all_contract_k_line(nkl.get_contract_li('OI'))
    # nkl.save_datas('FG', ['FG2009', 'FG2101'])
    # nkl.save_datas('AP', ['AP2105'])
    # run_datas_analyze()
    # nkl = ComposeNkLine()
    # nkl.get_nk_line('RB2101', win_n=5, pa=f'data_{5}m')




        


