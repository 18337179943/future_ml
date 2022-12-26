# %%
import imp
from io import SEEK_CUR
from operator import index, indexOf
import sys, os
from turtle import right
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
import re
from tkinter.messagebox import NO
from unicodedata import name
from charset_normalizer import from_path
from pyparsing import col
from rqdatac.services.live import current_snapshot
from rx import start
import talib as tb
import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
from copy import deepcopy
from numpy import ERR_CALL, abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import pymysql
import warnings
import rqdatac
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import pandas_ta as ta
from datetime import datetime, timedelta
# from m_base import *
import scipy.stats as st
import bottleneck as bn
# from datas_process.m_datas_process import BaseDataProcess
# from WindPy import w
# rqdatac.init('13574154200','123456')
rqdatac.init('18337179943','123456')
# df = rqdatac.get_price('RB88', start_date='20220701', end_date='20220706', frequency='60m')
# print(df)
# rqdatac.init('license','XNG6pTdiEs1-esY2cRvDWJ_y9bbExwI-GkEfLsoe9WVi-3s_MgMLeDEEY35V8CFqc0HdZ3dz0UPI_c_n7IxnA8o26jEJuphGRYEzDG8tUalEgnN7TQ1fZArrzKPEcujApKDcvDpJwbT4sZABGLa3bvgOJtXDyF7kFd3I1cXsJ3E=C6r3D2WhHm6-jCB3AnXLqpsHUs4-qkmvQWdHRLiINbrd9JfU-r0Qp15CapFPquDW_h4fnu9cUU9FJHfSlHs-iD2dq976erkoBHKBHmDCldQDaea6GbSDcStC66C-9SPL-eY2YTZ4JNQtqNfeoTihDyKVDvOIkOKbnUjUx8j4TYI=',("47.103.35.47",16010))
# w.start()
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示
plt.rcParams['axes.unicode_minus'] = False #解决符号无法显示

# %%
class SymbolsInfo():
    '''品种信息'''
    def __init__(self, contract_n=0):
        self.symbols_agricultural = [["A", 1, 10], ["B", 1, 10], ["M", 1, 10], ["RM", 1, 10], 
                        ["SR", 1, 10], ["CF", 5, 10], ["JD", 1, 5], ["C", 1, 10], ["CS", 1, 10], ["AP", 1, 10], ["CJ", 5, 5], ["RR", 1, 10],
                        ["RI", 1, 20], ["JR", 1, 20], ["LR", 1, 20], ["WH", 1, 20], ["PM", 1, 50]]  # 农产品
        self.symbols_chemical = [["EG", 1, 10], ["SC", 0.1, 1000], ["BU", 2, 10], ["SP", 2, 10], ["SA", 1, 20], ["FG", 1, 20], ["L", 1, 5], ["PP", 1, 5], ["EB", 1, 5],
                        ["UR", 1, 20], ["RU",5, 10], ["FU", 1, 10], ["BB", 0.05, 500], ["MA", 1, 10], ["FB", 0.5, 10], ["V", 1, 5], ["TA", 2, 5]]  # 能源和化工
        self.symbols_black = [["ZC", 0.2, 100], ["JM", 0.5, 60], ["J", 0.5, 100], ["I", 0.5, 100], ["RB", 1, 10], ["HC", 1, 10], ["SM", 2, 5], ["SF", 2, 5]]      # 黑色系
        self.symbols_metals = [["CU", 10, 5], ["AL", 5, 5], ["ZN", 5, 5], ["PB", 5, 5], ["NI", 10, 1], ["SN", 10, 1], ["AU", 0.02, 1000], ["AG", 1, 15]]     # 有色金属
        self.symbol_oil = [["OI", 1, 10], ["Y", 2, 10], ["P", 2, 10]]   # 油脂类
        self.symbols_CFFEX = [['IF', 0.2, 300], ['IC', 0.2, 200], ['IH', 0.2, 300]]
        self.CZCE_symbols = ['AP', 'WH', 'PM', 'CF', 'SR', 'OI', 'RI', 
                             'RS', 'RM', 'JR', 'LR', 'CY', 'CJ', 
                             'TA', 'MA', 'FG', 'ZC', 'SM', 'SF', 'UR', 'SA', 'PF']
        self.SHFE_symbols = ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG', 'RB',
                             'WR', 'HC', 'SS', 'SC', 'LU', 'FU', 'BU', 'RU', 'NR', 'SP']
        self.DCE_symbols = ['C', 'CS', 'A', 'B', 'M', 'Y', 'P', 'FB', 'BB', 'JD', 'RR', 
                            'L', 'V', 'PP', 'J', 'JM', 'I', 'EG', 'EB', 'PG']
        self.CFFEX_symbols = ['IC', 'IF', 'IH']
        self.INE_symbols = ['SC', 'NR', 'LU', 'BC']
        self.symbol_li = ['AP', 'AG', 'AL', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 
                                'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 
                                'SF', 'SN', 'SR', 'TA', 'V', 'Y', 'ZN']
        self.symbols_no_metals_li = ['AP', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 'I',  
                          'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 
                          'SC', 'SF', 'SR', 'TA', 'V', 'Y', 'ZC']
        self.symbol_li.sort()

        self.symbols_agricultural_li = ['AP', 'C', 'CF', 'CS', 'M', 'RM', 'JD', 'SR', 'CF']
        self.symbols_chemical_li = ['BU', 'RU', 'FG', 'PP', 'V', 'L', 'TA', 'MA']
        self.symbol_metals_li = ['AL', 'ZN', 'SN', 'CU', 'PB']
        self.symbols_black_li = ['RB', 'JM', 'J', 'SF', 'HC']
        self.symbol_oil_li = ['OI', 'Y', 'P']

        self.futures_name_li = ['agricultural', 'chemical', 'black', 'metals', 'oil']

        # self.symbol_li = ['AG', 'AL', 'AP', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 'IC', 'IF', 'IH', 
        #                         'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 
        #                         'SC', 'SF', 'SM', 'SN', 'SR', 'TA', 'V', 'Y', 'ZC', 'ZN']
        sector_c = [0]*len(self.symbols_agricultural) + [1]*len(self.symbols_chemical) + [2]*len(self.symbols_black) + \
                   [3]*len(self.symbols_metals) + [4]*len(self.symbols_CFFEX) + [5]*len(self.symbol_oil)

        self.symbols_all = self.symbols_agricultural + self.symbols_black + self.symbols_chemical + self.symbols_metals + self.symbols_CFFEX + self.symbol_oil
        self.np_symbols = np.array(self.symbols_all)
        self.symbols = list(self.np_symbols[:, 0])
        self.pricetick = list(self.np_symbols[:, 1])
        self.size = list(self.np_symbols[:, 2])

        self.symbols_classify = pd.DataFrame()

        self.symbols_classify['symbol'] = self.symbols
        self.symbols_classify['sector'] = sector_c

        self.df_symbols_all = pd.DataFrame(self.symbols_all)
        self.df_symbols_all.columns = ['symbol', 'pricetick', 'size']

        self.df_symbols_raw = self.df_symbols_all.copy()

        df_mc = pd.read_csv(f'{pa_prefix}/datas/df_symbol_chg_maincon.csv')
        price_li = []
        del_symbol = []

        for i in self.df_symbols_all['symbol']:
            try:
                price = df_mc[df_mc['object']==i]['close'].iloc[-1]
            except:
                price = 1000000
                del_symbol.append(i)
                # print('price is wrong', i)
            price_li.append(price)
        del_symbol += ['AU', 'SC', 'ZC']
        self.df_symbols_all['price'] = price_li 
        self.df_symbols_all = self.df_symbols_all[~self.df_symbols_all['symbol'].isin(del_symbol)]
        # self.df_symbols_all.to_csv('df_symbols_all.csv')
        if contract_n == 0:
            self.contract_rate = pd.read_excel(f'{pa_prefix}/datas/cntract_rate.xlsx')
            self.contract_rate[' 合约代码'] = self.contract_rate[' 合约代码'].apply(lambda x: x.upper())
        elif contract_n == 1:
            self.contract_rate = pd.read_csv(f'{pa_prefix}/datas/contract_rate_xz_adj.csv')
        self.symbols = self.df_symbols_all['symbol'].to_list()
    
    def dp_contract_rate_xz(self):
        '''对兴证期货手续费进行处理'''
        def set_contract(x: pd.DataFrame):
            symbol = get_sy(x[x['合约代码']!='!']['合约代码'].iloc[0])
            cost = np.max(x['每手金额'])
            if cost:
                rate0, rate1 = 0, 0
            else:
                rate0, rate1 = np.max(x['成交比例']), np.max(x['平今比例'])
            return {' 合约代码': [symbol], ' 开仓手续费(按手数)': [cost], 
                    ' 开仓手续费(按金额)': [rate0], ' 平今手续费(按手数)': [np.max(x['平今金额'])], 
                    ' 平今手续费(按金额)': [rate1]}
        # df = pd.read_csv(f'{pa_prefix}/datas/contract_rate_xz.csv')
        df = pd.read_excel(f'{pa_prefix}/datas/contract_rate_xz.xls')
        df = df[(df['投机/套保类型']=='!') & (df['合约代码']!='!') & (df['期权类型']!='!')]
        df['合约代码'] = df['合约代码'].apply(lambda x: get_sy(x).upper())
        df_res = pd.concat([pd.DataFrame(i) for i in df.groupby('合约类别').apply(set_contract)])
        df_res.to_csv(f'{pa_prefix}/datas/contract_rate_xz_adj.csv', index=False)
        return df_res
    
    def get_futures_li(self, futures_name='all'):
        '''提供期货板块品种'''
        return self.symbol_li if futures_name == 'all' else eval(f'self.symbols_{futures_name}_li')

    def get_size(self, symbol):
        '''获取合约乘数'''
        size = self.df_symbols_raw[self.df_symbols_raw['symbol']==symbol.upper()]['size'].iloc[0]
        return size

    def get_pricetick(self, symbol):
        '''获取合约最小变动价位'''
        pricetick = self.df_symbols_raw[self.df_symbols_raw['symbol']==symbol.upper()]['pricetick'].iloc[0]
        return pricetick

    def get_pricetick_rate_price(self):
        '''获取最小变动价/价格'''
        df = self.df_symbols_all.copy()
        df['tick_rate'] = df['pricetick'] / df['price']
        df.to_csv('df_symbol_info.csv')
        return df
    
    def get_rate(self, symbol):
        contract = self.contract_rate[self.contract_rate[' 合约代码']==symbol.upper()]
        symbol_info = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]
        price = symbol_info['price'].iloc[0]
        pricetick = symbol_info['pricetick'].iloc[0]
        size = symbol_info['size'].iloc[0]
        cr1 = contract[' 开仓手续费(按手数)'].iloc[0]
        cr2 = contract[' 开仓手续费(按金额)'].iloc[0]
        rate = cr2 if cr1 == 0 else cr1 / (price*size)
        return rate
    
    def get_rate_pricetick_size(self, symbol):
        '''获取合约参数'''
        contract = self.contract_rate[self.contract_rate[' 合约代码']==symbol.upper()]
        symbol_info = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]
        price = symbol_info['price'].iloc[0]
        pricetick = symbol_info['pricetick'].iloc[0]
        size = symbol_info['size'].iloc[0]
        cr1 = contract[' 开仓手续费(按手数)'].iloc[0]
        cr2 = contract[' 开仓手续费(按金额)'].iloc[0]
        rate = cr2 if cr1 == 0 else cr1 / (price*size)
        return rate, pricetick, size


class MainconInfo():
    '''每日主力合约信息'''
    def __init__(self, maincon_pa=f'{pa_prefix}/datas/maincon.csv'):
        self.maincon_pa = maincon_pa
        self.df_maincon = self.get_maincon(self.maincon_pa)
    
    def set_df_maincon(self, maincon_pa=f'{pa_prefix}/datas_sc/maincon.csv'):
        self.df_maincon = self.get_maincon(maincon_pa)

    def get_maincon(self, maincon_pa):
        '''获取每日主力合约'''
        df = pd.read_csv(maincon_pa)
        df['date'] = pd.to_datetime(df['date'])
        # df = df[(df['date'] >= self.startdate) & (df['date'] <= self.enddate)]
        return df 
    
    def maincon_startdate_enddate(self, contract, delay=0, delay_end=0):
        '''获取该合约主力时间段'''
        df = self.df_maincon[self.df_maincon['contract']==contract]
        startdate, enddate = df['date'].iloc[0], df['date'].iloc[-1]
        if delay: startdate = startdate - timedelta(days=delay)
        if delay_end: enddate = enddate + timedelta(days=delay_end)
        
        return startdate, enddate

    def get_symbol_df_maincon(self, symbol, startdate, enddate, delay=0, cut=1, delay_end=0):
        '''获取df: contract startdate enddate'''
        if not isinstance(startdate, datetime):
            startdate, enddate = str_to_datetime(startdate), str_to_datetime(enddate)

        df = self.df_maincon[self.df_maincon['symbol']==symbol.upper()]
        df = df[(df['date'].dt.date>=startdate.date()) & (df['date'].dt.date<=enddate.date())]
        contract_li = df['contract'].unique().tolist()
        startdate_li, enddate_li = [], []
        for contract in contract_li:
            st, en = self.maincon_startdate_enddate(contract, delay=delay, delay_end=delay_end)
            st, en = timestamp_to_datetime(st), timestamp_to_datetime(en)
            if st < startdate and cut: st = startdate
            if en > enddate: en = enddate
            startdate_li.append(st), enddate_li.append(en)
        df_res = pd.DataFrame({'contract': contract_li, 'startdate': startdate_li, 'enddate': enddate_li})
        df_res['symbol'] = symbol
        # print(df_res)
        # input()
        return df_res

    def get_df_contract(self, df_res: pd.DataFrame, load_pa=None, is_concat=0, contract_name=0, interval=60):
        '''获取主力合约时间段的k线'''
        df_li = []
        if load_pa is None:
            load_pa = f'{pa_prefix}/datas/data_{interval}m/'
        symbol = df_res.loc[0, 'symbol']
        for i in range(len(df_res)):
            contract, startdate, enddate, _ = df_res.iloc[i]
            df = pd.read_csv(f'{load_pa}{symbol}/{contract}.csv')
            df['datetime'] = pd.to_datetime(df['datetime'])
            # print(df['datetime'].iloc[0], df['datetime'].iloc[-1], startdate, enddate)
            df = df[(df['datetime']>startdate) & (df['datetime']<enddate)]

            if contract_name:
                df['contract'] = contract
            df_li.append(df.copy())

        if is_concat:
            df_concat = pd.concat(df_li)
            return df_concat

        return df_li
    
    def get_main_contact_k_line(self, symbol, startdate, enddate, delay=0, load_pa=None, is_concat=0, contract_name=0, interval=60, delay_end=0):
        '''获取主力合约时间段的k线'''
        df_res = self.get_symbol_df_maincon(symbol, startdate, enddate, delay, delay_end=delay_end)
        df_li = self.get_df_contract(df_res, load_pa, is_concat, contract_name=contract_name, interval=interval)
        return df_li



class LoadFactors():
    '''下载因子，每个期货品种一张表'''
    def __init__(self, startdate='20130401', enddate='20211101', interval="1d"):
        self.startdate = startdate
        self.enddate = enddate
        self.interval = interval
        self.symbols_class = SymbolsInfo()
        self.df_symbols = self.symbols_class.df_symbols
        self.symbols_agricultural = self.symbols_class.symbols_agricultural
        self.symbols_Chemical = self.symbols_class.symbols_Chemical
        self.symbols_black = self.symbols_class.symbols_black
        self.symbols_metals = self.symbols_class.symbols_metals
        self.symbols = self.symbols_class.symbols
        self.symbols_list = self.symbols_class.symbols_list
        self.path = self.symbols_class.path
        self.time_step = [[startdate, '20140901'], ['20140902', '20150801'], ['20150802', '20160701'], 
                          ['20160702', '20170601'], ['20170602', '20180501'], ['20180502', '20190401'],  
                          ['20190402', '20200301'], ['20200302', '20210201'], ['20210202', enddate]]

    def get_cursor(self):
        '''获取通联接口'''
        connect = pymysql.connect(host='rm-uf68ez0445d101yimvo.mysql.rds.aliyuncs.com',
                                port=3306,
                                db='tldata_db',
                                user='liunian',
                                password='IsFlo5R82lhdwMtmu24Y')
        cursor = connect.cursor()
        return cursor

    def load_datas(self, sql, cursor):
        '''执行sql语句下载数据'''
        cursor.execute(sql)
        data = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        return df

    def get_path(self, symbol):
        '''获取存储路径'''
        if symbol in self.symbols_agricultural:
            return self.path[0]
        elif symbol in self.symbols_black:
            return self.path[1]
        elif symbol in self.symbols_Chemical:
            return self.path[2]
        else:
            return self.path[3]

    def get_rqdatas(self, symbol='SF'):
        df = rqdatac.get_price(symbol, start_date=self.startdate, end_date=self.enddate, frequency=self.interval)
        df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'open_interest', 'settlement', 'prev_settlement']]
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest', 'settlement', 'prev_settlement']
        return df

    def get_datayes(self, symbol='SF'):
        '''下载通联数据'''
        sql = '''
        select TRADE_DATE date, OPEN_PRICE open, HIGHEST_PRICE high, LOWEST_PRICE low, CLOSE_PRICE close, 
        TURNOVER_VOL volume, TURNOVER_VALUE turnover, OPEN_INT, CHG, CHG1, CHG_PCT,
        PRE_SETTL_PRICE, SETTL_PRICE, PRE_CLOSE_PRICE pre_close 
        from mkt_futd 
        where CONTRACT_OBJECT='{}' and TRADE_DATE between {} and {} and MAINCON=1 
        order by TRADE_DATE'''.format(symbol, self.startdate, self.enddate)
        cursor = self.get_cursor()
        df = self.load_datas(sql, cursor)
        df[['open', 'high', 'low', 'close', 'SETTL_PRICE']] = df[['open', 'high', 'low', 'close', 'SETTL_PRICE']].fillna(method='bfill', axis=1)
        df = df.fillna(method='ffill', axis=0)
        df.dropna(inplace=True)
        df.set_index('date', inplace=True)
        df = df.apply(m_format)
        # df[['open', 'h']]
        return df
    
    def get_wind(self, symbol):
        '''获取wind数据'''
        df_list = []
        for dt in self.time_step:
            df0 = w.wset("productsfund","startdate={};enddate={};product={}; \
            field=date,holdnumber,holdnumberchange,holdnumber_margin,holdnumberchange_margin".format(dt[0], dt[1], symbol), usedf=True)[1]
            if len(df0) != 0:
                print(df0)
                df0.set_index('date', inplace=True)
                df0.sort_index(inplace=True)        
                symbol_exchange = self.df_symbols[self.df_symbols['symbol']==symbol]['index'].iloc[0]
                exchange = self.df_symbols[self.df_symbols['symbol']==symbol]['exchange'].iloc[0]
                exchange = 'CZCE' if exchange == 'CZC' else exchange
                df1 = w.wsd(symbol_exchange, "st_stock,anal_basis,anal_basispercent2", dt[0], dt[1], "TradingCalendar={};Currency=CNY".format(exchange), usedf=True)[1]
                df_i = pd.merge(df0, df1, left_index=True, right_index=True)
                df_i.fillna(0, inplace=True)
                df_list.append(df_i.copy())
        df = pd.concat(df_list)
        return df

    def get_factors(self, symbol):
        '''获取和保存因子'''
        try:
            df = self.get_datayes(symbol[:-2])
            # df = self.datayes.get_price(symbol[:-2])
            # df = self.get_rqdatas(symbol)
            # df = get_datayes(symbol, start_date=self.startdate, end_date=self.enddate, frequency=self.interval)
            df = get_alpha(df)
            df = get_talib(df)
            df = get_mf(df)
            df = df.iloc[65:]
            df.fillna(0, inplace=True)
            # df_w = self.get_wind(symbol[:-2])
            # df = pd.merge(df_w, df, left_index=True, right_index=True)
            path = self.get_path(symbol)
            df.to_csv(path+symbol[:-2]+".csv")
            return {symbol: 1}
        except:
            return {symbol: 2}

    def factor_example(self):
        '''因子样本'''
        symbol = "ZC88"
        # df = self.get_datayes(symbol[:-2])
        # df = self.datayes.get_price(symbol[:-2])
        df = rqdatac.get_price(symbol, start_date=self.startdate, end_date=self.enddate, frequency=self.interval)
        df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'open_interest', 'settlement', 'prev_settlement']]
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest', 'settlement', 'prev_settlement']
        # df.columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        # df = get_datayes(symbol, start_date=self.startdate, end_date=self.enddate, frequency=self.interval)
        df = get_alpha(df)
        df = get_talib(df)
        df = df.iloc[65:]
        df.fillna(0, inplace=True)
        # df_w = self.get_wind(symbol[:-2])
        # df = pd.merge(df_w, df, left_index=True, right_index=True)
        path = self.get_path(symbol)
        df.to_csv(path+symbol[:-2]+".csv")
        return df

    def multi_progress_save_factors(self):
        '''多进程下载因子数据'''
        with ProcessPoolExecutor(max_workers=7) as executor:  # max_workers=10
            results = executor.map(self.get_factors, self.symbols)
        res = pd.DataFrame([r for r in results])
        return res

    def normalize(self, x):
        '''数据标准化'''
        if len(pd.value_counts(x)) > 3:
            x_train = x.loc[x.index <= '2020-12-31']
            x = (x - x_train.mean()) / x_train.std()
        return x

    def datas_process(self, industry_index=1):
        '''数据处理'''
        df_all = []
        pa = self.path[industry_index]
        # 获取某个板块对应期货品种的数据
        for symbol in self.symbols_list[industry_index]:
            df = pd.read_csv(pa+symbol+".csv")
            df.set_index('date', inplace=True)
            df['rate'] = (df['open'].shift(-3) - df['open'].shift(-1)) / df['open'].shift(-1)
            df.iloc[:, :-1] = df.iloc[:, :-1].apply(self.normalize)
            df['symbol'] = symbol
            df.reset_index(inplace=True)
            df.set_index(['date','symbol'], inplace=True)
            df_all.append(df)

        df = pd.DataFrame()
        for i in range(len(df_all)):
            df = pd.concat([df, df_all[i]])
        df.sort_index(inplace=True, ascending=True)
        df['y'] = df.groupby(df.index.get_level_values(level = 0))['rate'].apply(pd.qcut,3,False,False,3,'drop')
        del df['rate']
        df.dropna(inplace=True)
        return df


class DataYes():
    def __init__(self, start_date='20140101', end_date='20210801'):
        # self.symbol = symbol
        # self.symbol = ['SF', 'MA', 'OI', 'P', 'PB', 'PP', 'RB', 'RI', 'RM', 'RS', 'RU', 'M', 'SM', 'SR', 'TA', 'ZC', 'TF', 'V', 'WH', 'WR', 'Y', 'ZN', 'FG', 'AG', 'AL', 'AU', 'B', 'BB', 'BU', 'C', 'CF', 'CU', 'FB', 'A', 'FU', 'HC', 'I', 'J', 'JD', 'JM', 'L', 'LR']
        self.start_date = start_date
        self.end_date = end_date
        self.sql_list = []
    
    def sql_language(self):
        # sql1 = 
        pass

    def get_price(self, symbol):
        sql = '''
        select TRADE_DATE date, OPEN_PRICE open, HIGHEST_PRICE high, LOWEST_PRICE low, CLOSE_PRICE close, 
        TURNOVER_VOL volume, TURNOVER_VALUE turnover, OPEN_INT, CHG, CHG1, CHG_PCT
        PRE_SETTL_PRICE, PRE_CLOSE_PRICE, SETTL_PRICE, PRE_CLOSE_PRICE pre_close 
        from mkt_futd 
        where CONTRACT_OBJECT='{}' and TRADE_DATE between {} and {} and MAINCON=1 
        order by TRADE_DATE'''.format(symbol, self.start_date, self.end_date)
        cursor = get_cursor()
        df = load_datas(sql, cursor)

        return df


def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

# def get_yearly_autocorr(data):
    # ac = acf(data, nlags=366)
    # return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])

def transform_date(data):
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['day'] = data['date'].dt.day
    data['year_mod'] = (data['year'] - data['year'].min()) / \
        (data['year'].max() - data['year'].min())
    data['dayofweek_sin'] = sin_transform(data['dayofweek'])
    data['dayofweek_cos'] = cos_transform(data['dayofweek'])
    data['month_sin'] = sin_transform(data['month'])
    data['month_cos'] = cos_transform(data['month'])
    data['day_sin'] = sin_transform(data['day'])
    data['day_cos'] = cos_transform(data['day'])
    # data['yearly_autocorr'] = get_yearly_autocorr(data['date'])
    data['diff'] = (data['date'] - data['date'].shift(1)).apply(lambda x: x.days)
    data['diff'] = data['diff'].fillna(1)
    data = data.set_index('date')
    return data

# def add_time_factor(factor):
#     time_factor = transform_date(pd.DataFrame(pd.Series(
#         factor.index.get_level_values(0).unique()).sort_values(), columns=['date']))
#     # 时间间隔因子
#     return time_factor

def add_time_factor(factor: pd.DataFrame):
    '''时间因子'''
    df = factor.copy()
    df = df.reset_index()
    df['month'] = df['datetime'].apply(lambda x: x.month)
    df['day'] = df['datetime'].apply(lambda x: x.day)
    df['hour'] = df['datetime'].apply(lambda x: x.month)
    df['minute'] = df['datetime'].apply(lambda x: x.month)
    df['month_sin'] = sin_transform(df['month'])
    df['day_sin'] =  sin_transform(df['day'])
    df['hour_sin'] = sin_transform(df['hour'])
    df['minute_sin'] = sin_transform(df['minute'])
    df['month_cos'] = cos_transform(df['month'])
    df['day_cos'] =  cos_transform(df['day'])
    df['hour_cos'] = cos_transform(df['hour'])
    df['minute_cos'] = cos_transform(df['minute'])
    df = df.set_index('datetime')
    return df

def get_mf(df):
    '''自己写的指标'''
    df = return_rate(df, 5)
    df = return_rate(df, 22)
    return df
    
def plot_hist(series, title_name, pa):
    df = pd.DataFrame(series)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    plt.hist(df.iloc[:, 0], bins=1000)
    plt.title(title_name)
    plt.savefig(pa+title_name+'.png')
    plt.close()

def transform_factors():
    '''转换因子，将每个期货品种一张表转成每个因子一张表'''
    symbolsinfo = SymbolsInfo()
    df_dir = {}
    for path in symbolsinfo.path:  # 加载数据表
        table_list = os.listdir(path)
        for table in table_list:
            df = pd.read_csv(path+table)
            df.set_index('date', inplace=True)
            df_dir[table[:-4]] = df
    for factor_name in df.columns.tolist():
        df_factor = pd.DataFrame({i : df_dir[i][factor_name] for i in df_dir})
        df_factor.to_csv('{pa_prefix}/factors/'+factor_name+'.csv') 

def return_rate(df, n):
    '''计算n周期的收益率 5/22'''
    df['returnrate'+str(n)] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    return df

def seq_momentum_signal(df):
    '''计算时间序列动量因子多空信号'''
    df = df.apply(lambda x: np.sign(x))
    return df

def cross_momentum(df: pd.DataFrame):
    '''计算横截面动量因子多空信号'''
    def signal(x):
        x = pd.qcut(x, 5, labels=False)
        x[(x != 0) & (x != 4)] = 5
        x[x==0] = -1
        x[x==4] = 1
        x[(x != 0) & (x != 4)] = 0
        return x
    df = df.apply(signal, axis=1)
    return df

def get_cursor():
    connect = pymysql.connect(host='rm-uf68ez0445d101yimvo.mysql.rds.aliyuncs.com',
                              port=3306,
                              db='tldata_db',
                              user='liunian',
                              password='IsFlo5R82lhdwMtmu24Y')
    cursor = connect.cursor()
    return cursor

def load_datas(sql, cursor):
    cursor.execute(sql)
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    return df


def get_index(df):
    df = get_alpha(df)
    df = get_talib(df)
    return df

def get_rqdatas(symbol, startdate, enddate, interval):
    '''获取k线数据'''
    df = rqdatac.get_price(symbol, start_date=startdate, end_date=enddate, frequency=interval)
    df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover']]
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
    return df



# region Auxiliary functions
def m_format(x):
    x = x.apply(lambda x: float(x))
    return x

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    #return df.rank(axis=1, pct=True)
    return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :] 
    # na_series = df.as_matrix()
    na_series = df.values

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  
# endregion

def get_alpha0(df):
    # df_copy = df.copy()
    stock=Alphas(df)
    # df = {}
    df['alpha001']=stock.alpha001() 
    df['alpha002']=stock.alpha002()
    df['alpha003']=stock.alpha003()
    df['alpha004']=stock.alpha004()
    df['alpha005']=stock.alpha005()
    df['alpha006']=stock.alpha006()
    df['alpha007']=stock.alpha007()
    df['alpha008']=stock.alpha008()
    df['alpha009']=stock.alpha009()
    df['alpha010']=stock.alpha010()
    df['alpha011']=stock.alpha011()
    df['alpha012']=stock.alpha012()
    df['alpha013']=stock.alpha013()
    df['alpha014']=stock.alpha014()
    df['alpha015']=stock.alpha015()
    df['alpha016']=stock.alpha016()
    df['alpha017']=stock.alpha017()
    df['alpha018']=stock.alpha018()
    df['alpha019']=stock.alpha019()
    df['alpha020']=stock.alpha020()
    df['alpha021']=stock.alpha021()
    df['alpha022']=stock.alpha022()
    df['alpha023']=stock.alpha023()
    df['alpha024']=stock.alpha024()
    df['alpha025']=stock.alpha025()
    df['alpha026']=stock.alpha026()
    # df['alpha027']=stock.alpha027()
    df['alpha028']=stock.alpha028()
    df['alpha029']=stock.alpha029()
    df['alpha030']=stock.alpha030()
    df['alpha031']=stock.alpha031()
    df['alpha032']=stock.alpha032()
    df['alpha033']=stock.alpha033()
    df['alpha034']=stock.alpha034()
    df['alpha035']=stock.alpha035()
    df['alpha036']=stock.alpha036()
    df['alpha037']=stock.alpha037()
    df['alpha038']=stock.alpha038()
    df['alpha039']=stock.alpha039()
    df['alpha040']=stock.alpha040()
    df['alpha041']=stock.alpha041()
    df['alpha042']=stock.alpha042()
    df['alpha043']=stock.alpha043()
    df['alpha044']=stock.alpha044()
    df['alpha045']=stock.alpha045()
    df['alpha046']=stock.alpha046()
    df['alpha047']=stock.alpha047()
    df['alpha049']=stock.alpha049()
    df['alpha050']=stock.alpha050()
    df['alpha051']=stock.alpha051()
    df['alpha052']=stock.alpha052()
    df['alpha053']=stock.alpha053()
    df['alpha054']=stock.alpha054()
    df['alpha055']=stock.alpha055()
    df['alpha057']=stock.alpha057()
    df['alpha060']=stock.alpha060()
    df['alpha061']=stock.alpha061()
    df['alpha062']=stock.alpha062()
    df['alpha064']=stock.alpha064()
    df['alpha065']=stock.alpha065()
    df['alpha066']=stock.alpha066()
    df['alpha068']=stock.alpha068()
    df['alpha071']=stock.alpha071()
    df['alpha072']=stock.alpha072()
    df['alpha073']=stock.alpha073()
    df['alpha074']=stock.alpha074()
    df['alpha075']=stock.alpha075()
    df['alpha077']=stock.alpha077()
    df['alpha078']=stock.alpha078()
    df['alpha081']=stock.alpha081()
    df['alpha083']=stock.alpha083()
    # df['alpha084']=stock.alpha084()
    df['alpha085']=stock.alpha085()
    df['alpha086']=stock.alpha086()
    df['alpha088']=stock.alpha088()
    df['alpha092']=stock.alpha092()
    df['alpha094']=stock.alpha094()
    df['alpha095']=stock.alpha095()
    df['alpha096']=stock.alpha096()
    df['alpha098']=stock.alpha098()
    df['alpha099']=stock.alpha099()
    df['alpha101']=stock.alpha101() 
    # print(df)
    # df = pd.DataFrame(df)
    # df.index = df_copy.index
    # df = pd.merge(df_copy, df, left_index=True, right_index=True) 
    return df
    
def get_talib(df):
    # df_copy = df.copy()
    tbindex = talib_index(df)
    # df = {}
    # df['sma'] = tbindex.sma()
    # df['ema'] = tbindex.ema()
    # df['kama'] = tbindex.kama()
    # df['wma'] = tbindex.wma()
    # df['dema'] = tbindex.dema()
    # df['mama1'], df['mama2'] = tbindex.mama()
    df['ht_trendline'] = tbindex.ht_trendline()
    df['minpoint'] = tbindex.minpoint()
    df['midprice'] = tbindex.midprice()
    df['sar'] = tbindex.sar()
    df['sarext'] = tbindex.sarext()
    df['t3'] = tbindex.t3()
    df['trima'] = tbindex.trima()
    df['macdext1'], df['macdext2'], df['macdext3'] = tbindex.macdext()
    df['macdfix1'], df['macdfix2'], df['macdfix3'] = tbindex.macdfix()
    df['mfi'] = tbindex.mfi()
    df['cmo'] = tbindex.cmo()
    df['mom'] = tbindex.mom()
    df['ppo'] = tbindex.ppo()
    df['roc'] = tbindex.roc()
    df['rocr'] = tbindex.rocr()
    df['rocp'] = tbindex.rocp()
    df['rocr_100'] = tbindex.rocr_100()
    df['trix'] = tbindex.trix()
    df['std'] = tbindex.std()
    df['obv'] = tbindex.obv()
    df['cci'] = tbindex.cci()
    df['atr'] = tbindex.atr()
    df['natr'] = tbindex.natr()
    df['rsi'] = tbindex.rsi()
    df['stoch1'], df['stoch2'] = tbindex.stoch()
    df['stochf1'], df['stochf2'], = tbindex.stochf()
    df['stochrsi1'], df['stochrsi2'] = tbindex.stochrsi()
    df['macd1'], df['macd2'], df['macd3'] = tbindex.macd()
    df['adx'] = tbindex.adx()
    df['adxr'] = tbindex.adxr()
    df['dx'] = tbindex.dx()
    df['minus_di'] = tbindex.minus_di()
    df['plus_di'] = tbindex.plus_di()
    df['willr'] = tbindex.willr()
    df['ultosc'] = tbindex.ultosc()
    df['trange'] = tbindex.trange()
    df['bollup'], df['bolldown'] = tbindex.boll()
    df['keltnerup'], df['keltnerdown'] = tbindex.keltner()
    df['avgprice'] = tbindex.avgprice()
    df['medprice'] = tbindex.medprice()
    df['typprice'] = tbindex.typprice()
    df['wclprice'] = tbindex.wclprice()
    df['add'] = tbindex.add()
    df['div'] = tbindex.div()
    df['max'] = tbindex.max()
    df['maxindex'] = tbindex.maxindex()
    df['min'] = tbindex.min()
    df['minindex'] = tbindex.minindex()
    df['mult'] = tbindex.mult()
    df['sub'] = tbindex.sub()
    df['sum'] = tbindex.sum()
    df['ht_dcperiod'] = tbindex.ht_dcperiod()
    df['ht_dcphase'] = tbindex.ht_dcphase()
    df['ht_phasor1'], df['ht_phasor2'] = tbindex.ht_phasor()
    df['ht_sine1'], df['ht_sine2'] = tbindex.ht_sine()
    df['ht_trendmode'] = tbindex.ht_trendmode()
    df['cdl2crows'] = tbindex.cdl2crows()
    df['cdl3blackcrows'] = tbindex.cdl3blackcrows()
    df['cdl3inside'] = tbindex.cdl3inside()
    df['cdl3instrike'] = tbindex.cdl3instrike()
    df['cdl3outside'] = tbindex.cdl3outside()
    df['cdl3starsinsouth'] = tbindex.cdl3starsinsouth()
    df['cdl3whitsoldiers'] = tbindex.cdl3whitsoldiers()
    df['cdabandonbaby'] = tbindex.cdabandonbaby()
    df['cdbelthold'] = tbindex.cdbelthold()
    df['cdbelthold'] = tbindex.cdbelthold()
    df['cdadvanceblock'] = tbindex.cdadvanceblock()
    df['cdbreakaway'] = tbindex.cdbreakaway()
    df['cdclosingmarubozu'] = tbindex.cdclosingmarubozu()
    df['cdconcealingbabyswallow'] = tbindex.cdconcealingbabyswallow()
    df['cdleveningdojistar'] = tbindex.cdleveningdojistar()
    df['cdleveingstar'] = tbindex.cdleveingstar()
    df['cdldojistar'] = tbindex.cdldojistar()
    df['cdlgapsidewhite'] = tbindex.cdlgapsidewhite()
    df['cdlggravestonedoji'] = tbindex.cdlggravestonedoji()
    df['cdlhammer'] = tbindex.cdlhammer()
    df['cdlhangingman'] = tbindex.cdlhangingman()
    df['cdlharami'] = tbindex.cdlharami()
    df['cdlharamicross'] = tbindex.cdlharamicross()
    df['cdlhighwave'] = tbindex.cdlhighwave()
    df['cdlhikkake'] = tbindex.cdlhikkake()
    df['cdlhikkakemode'] = tbindex.cdlhikkakemode()
    df['cdlhomingpigeon'] = tbindex.cdlhomingpigeon()
    df['cdlidentical3crows'] = tbindex.cdlidentical3crows()
    df['sdlinneck'] = tbindex.sdlinneck()
    df['cdlinvertedhammer'] = tbindex.cdlinvertedhammer()
    df['cdlkicking'] = tbindex.cdlkicking()
    df['cdlkickingbylength'] = tbindex.cdlkickingbylength()
    df['cdlladderbottom'] = tbindex.cdlladderbottom()
    df['cdllongleggeddoji'] = tbindex.cdllongleggeddoji()
    df['cdllongline'] = tbindex.cdllongline()
    df['cdlmarubozu'] = tbindex.cdlmarubozu()
    df['cdlmatchhighlow'] = tbindex.cdlmatchhighlow()
    df['cdlmathold'] = tbindex.cdlmathold()
    df['cdlmorningdojistar'] = tbindex.cdlmorningdojistar()
    df['cdlmorningstar'] = tbindex.cdlmorningstar()
    df['cdlonneck'] = tbindex.cdlonneck()
    df['cdlpiercing'] = tbindex.cdlpiercing()
    df['cdlrickshawman'] = tbindex.cdlrickshawman()
    df['cdlrisefall3methods'] = tbindex.cdlrisefall3methods()
    df['cdlseparatinglines'] = tbindex.cdlseparatinglines()
    df['cdlshootingstar'] = tbindex.cdlshootingstar()
    df['cdlshortline'] = tbindex.cdlshortline()
    df['cdlspinningtop'] = tbindex.cdlspinningtop()
    df['cdlstalledpattern'] = tbindex.cdlstalledpattern()
    df['cdlsticksandwich'] = tbindex.cdlsticksandwich()
    df['cdltankurl'] = tbindex.cdltankurl()
    df['cdltasukigap'] = tbindex.cdltasukigap()
    df['cdlthrusting'] = tbindex.cdlthrusting()
    df['cdltristar'] = tbindex.cdltristar()
    df['cdlunique3river'] = tbindex.cdlunique3river()
    df['cdlupsidegap2crows'] = tbindex.cdlupsidegap2crows()
    df['cdlxssidegap3methods'] = tbindex.cdlxssidegap3methods()
    df['beta'] = tbindex.beta()
    df['correl'] = tbindex.correl()
    df['linearreg'] = tbindex.linearreg()
    df['linerreg_angle'] = tbindex.linerreg_angle()
    df['linearreg_intercept'] = tbindex.linearreg_intercept()
    df['linearreg_slope'] = tbindex.linearreg_slope()
    df['tsf'] = tbindex.tsf()
    df['var'] = tbindex.var()
    df['atan'] = tbindex.atan()
    df['ceil'] = tbindex.ceil()
    df['cos'] = tbindex.cos()
    df['floor'] = tbindex.floor()
    df['ln'] = tbindex.ln()
    df['log10'] = tbindex.log10()
    df['aroonup'], df['aroondown'] = tbindex.aroon()
    df['aroonosc'] = tbindex.aroonosc()
    df['ad'] = tbindex.ad()
    df['adosc'] = tbindex.adosc()
    df['bop'] = tbindex.bop()
    df['minus_dm'] = tbindex.minus_dm()
    df['plus_dm'] = tbindex.plus_dm()
    df = pd.DataFrame(df)
    df.dropna(axis=1, how='all')
    # df = pd.DataFrame(df)
    # df.index = df_copy.index
    # df = pd.merge(df_copy, df, left_index=True, right_index=True) 
    return df

def get_alpha_one(df, i):
        stock=Alphas(df)
        df = {}
        df['alpha001']=stock.alpha001
        df['alpha002']=stock.alpha002
        df['alpha003']=stock.alpha003
        df['alpha004']=stock.alpha004
        # df['alpha005']=stock.alpha005()
        df['alpha006']=stock.alpha006
        df['alpha007']=stock.alpha007
        df['alpha008']=stock.alpha008
        df['alpha009']=stock.alpha009
        df['alpha010']=stock.alpha010
        # df['alpha011']=stock.alpha011()
        df['alpha012']=stock.alpha012
        df['alpha013']=stock.alpha013
        df['alpha014']=stock.alpha014
        df['alpha015']=stock.alpha015
        df['alpha016']=stock.alpha016
        df['alpha017']=stock.alpha017
        df['alpha018']=stock.alpha018
        # df['alpha019']=stock.alpha019()
        df['alpha020']=stock.alpha020
        df['alpha021']=stock.alpha021
        df['alpha022']=stock.alpha022
        df['alpha023']=stock.alpha023
        df['alpha024']=stock.alpha024
        # df['alpha025']=stock.alpha025()
        df['alpha026']=stock.alpha026
        # df['alpha027']=stock.alpha027()
        df['alpha028']=stock.alpha028
        df['alpha029']=stock.alpha029
        df['alpha030']=stock.alpha030
        df['alpha031']=stock.alpha031
        # df['alpha032']=stock.alpha032()
        df['alpha033']=stock.alpha033
        df['alpha034']=stock.alpha034
        df['alpha035']=stock.alpha035
        df['alpha036']=stock.alpha036
        df['alpha037']=stock.alpha037
        df['alpha038']=stock.alpha038
        # df['alpha039']=stock.alpha039()
        df['alpha040']=stock.alpha040
        # df['alpha041']=stock.alpha041()
        # df['alpha042']=stock.alpha042()
        df['alpha043']=stock.alpha043
        df['alpha044']=stock.alpha044
        df['alpha045']=stock.alpha045
        df['alpha046']=stock.alpha046
        # df['alpha047']=stock.alpha047()
        df['alpha049']=stock.alpha049
        # df['alpha050']=stock.alpha050()
        df['alpha051']=stock.alpha051
        # df['alpha052']=stock.alpha052()
        df['alpha053']=stock.alpha053
        df['alpha054']=stock.alpha054
        df['alpha055']=stock.alpha055
        df['alpha057']=stock.alpha057
        df['alpha060']=stock.alpha060
        # df['alpha061']=stock.alpha061()
        # df['alpha062']=stock.alpha062()
        df['alpha064']=stock.alpha064
        # df['alpha065']=stock.alpha065()
        # df['alpha066']=stock.alpha066()
        # df['alpha068']=stock.alpha068()
        # df['alpha071']=stock.alpha071()
        df['alpha072']=stock.alpha072
        # df['alpha073']=stock.alpha073()
        # df['alpha074']=stock.alpha074()
        # df['alpha075']=stock.alpha075()
        # df['alpha077']=stock.alpha077()
        df['alpha078']=stock.alpha078
        # df['alpha081']=stock.alpha081()
        # df['alpha083']=stock.alpha083()
        # df['alpha084']=stock.alpha084()
        df['alpha085']=stock.alpha085
        # df['alpha086']=stock.alpha086()
        df['alpha088']=stock.alpha088
        df['alpha092']=stock.alpha092
        # df['alpha094']=stock.alpha094()
        # df['alpha095']=stock.alpha095()
        # df['alpha096']=stock.alpha096()
        # df['alpha098']=stock.alpha098()
        df['alpha099']=stock.alpha099
        df['alpha101']=stock.alpha101
        if i < 10:
            return df['alpha00'+str(i)]()
        else:
            try:
                return df['alpha0'+str(i)]()
            except:
                return df['alpha0'+str(18)]()

def get_alpha(df):
    # df_copy = df.copy()
    stock=Alphas(df)
    # df = {}
    df['alpha006']=stock.alpha006()
    df['alpha007']=stock.alpha007()
    df['alpha009']=stock.alpha009()
    df['alpha010']=stock.alpha010()
    df['alpha012']=stock.alpha012()
    df['alpha021']=stock.alpha021()
    df['alpha023']=stock.alpha023()
    df['alpha024']=stock.alpha024()
    df['alpha028']=stock.alpha028()
    df['alpha032']=stock.alpha032()
    df['alpha046']=stock.alpha046()
    df['alpha049']=stock.alpha049()
    df['alpha051']=stock.alpha051()
    df['alpha053']=stock.alpha053()
    df['alpha054']=stock.alpha054()
    df['alpha096']=stock.alpha096()
    df['alpha101']=stock.alpha101() 
    # print(df)
    # df = pd.DataFrame(df)
    # df.index = df_copy.index
    # df = pd.merge(df_copy, df, left_index=True, right_index=True) 
    return df

def get_shift(df, shift_n1, shift_n2=None):
    df1 = df.copy()
    for i in df1.columns:
        df1[i+str(shift_n1)] = df[i].shift(shift_n1)
        if shift_n2 != None:
            df1[i+str(shift_n2)] = df[i].shift(shift_n2)
    return df1

def get_pandas_ta(df, n):
    '''获取pandas_ta'''
    df1 = df.copy()
    
    s = ['aberration', 'above', 'above_value', 'accbands', 'ad', 'adx', 
        'alma', 'amat', 'ao', 'aobv', 'apo', 'aroon', 'atr', 'bbands', 'below', 
        'below_value', 'bias', 'bop', 'brar', 'cci', 'cdl_pattern', 'cdl_z', 
        'cfo', 'cg', 'chop', 'cksp', 'cmf', 'cmo', 'coppock', 'cross', 'cross_value', 
        'cti', 'decay', 'decreasing', 'dema', 'dm', 'donchian', 'dpo', 'ebsw', 'efi',
        'ema', 'entropy', 'eom', 'er', 'eri', 'fisher', 'fwma', 'ha', 'hilo', 'hl2', 
        'hlc3', 'hma', 'hwma', 'ichimoku', 'increasing', 'inertia', 'jma', 
        'kama', 'kc', 'kdj', 'kst', 'kurtosis', 'kvo', 'linreg', 'log_return', 'long_run', 
        'macd', 'mad', 'massi', 'mcgd', 'median', 'mfi', 'midpoint', 'midprice', 'mom', 
        'natr', 'nvi', 'obv', 'ohlc4', 'pdist', 'percent_return', 'pgo', 'ppo', 
        'psar', 'psl', 'pvi', 'pvo', 'pvol', 'pvr', 'pvt', 'pwma', 'qqe', 'qstick', 
        'quantile', 'rma', 'roc', 'rsi', 'rsx', 'rvgi', 'rvi', 'short_run', 'sinwma', 
        'skew', 'slope', 'sma', 'smi', 'squeeze', 'squeeze_pro', 'ssf', 'stc', 'stdev', 
        'stoch', 'stochrsi', 'supertrend', 'swma', 't3', 'td_seq', 'tema', 'thermo', 
        'tos_stdevall', 'trima', 'trix', 'true_range', 'tsi', 'tsignals', 'ttm_trend', 
        'ui', 'uo', 'variance', 'vhf', 'vidya', 'vortex', 'vp', 'vwap', 'vwma', 'wcp', 
        'willr', 'wma', 'xsignals', 'zlma', 'zscore']

    for i in s:
        # df_i = getattr(df1.ta, i)(n)
        # df1[df_i.columns.to_list()] = df_i
        try:
            df_i = getattr(df1.ta, i)(n)
            df1[df_i.columns.to_list()] = df_i
        except:
            pass
     
    return df1
    


class Alphas(object):
    def __init__(self, df_data):
        df_data = df_data.copy()
        self.open = df_data['open']
        self.high = df_data['high']
        self.low = df_data['low']
        self.close = df_data['close']
        self.volume = df_data['volume']
        self.returns = df_data['close'] - df_data['close'].shift(1)
        self.vwap = df_data['turnover']/df_data['volume']  # [5 11 25 27 32 41 42 47 50 61 62 65 66 71 73 74 75 77 81 83 84 94 96 98]
        
    # Alpha#1	 (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha001(self):
        inner = self.close
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5))
    
    # Alpha#2	 (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)
    
    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        return  (rank((self.open - (sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.close - self.vwap)))))
    
    # Alpha#6	 (-1 * correlation(open, volume, 10))
    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha
    
    # Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
    
    # Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    # Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    def alpha011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) *rank(delta(self.volume, 3)))
    
    # Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))
    
    # Alpha#14	 ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(delta(self.returns, 3)) * df
    
    # Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)
    
    # Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))
    
    # Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))
        
    # Alpha#18	 (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                          df))
    
    # Alpha#19	 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 25))))  # 250
    
    # Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index
                             )
    #        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
    #                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha
    
    # Alpha#22	 (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close),index=self.close.index,columns=['close'])
        alpha.at[cond,'close'] = -1 * delta(self.high, 2).fillna(value=0)
        return alpha
    
    # Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha
    
    # Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))
    
    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)
    
    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    ###
    ## Some Error, still fixing!!
    def alpha027(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5]=1
        return alpha  
    
    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    # Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)         
        p1=rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))).to_frame(), 10)))) 
        p2=rank((-1 * delta(self.close, 3)))
        p3=sign(scale(df))
        
        return p1.CLOSE+p2+p3

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha032(self):
        return scale(((sma(self.close, 7) / 7) - self.close)) + (20 * scale(correlation(self.vwap, delay(self.close, 5),23)))
    
    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        return rank(-1 + (self.open / self.close))
    
    # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))
            
    # Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha036(self):
        adv20 = sma(self.volume, 20)
        return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (0.7 * rank((self.open- self.close)))) + (0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(abs(correlation(self.vwap,adv20, 6)))) + (0.6 * rank((((sma(self.close, 30) / 30) - self.open) * (self.close - self.open)))))
    
    # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 30)) + rank(self.open - self.close) # 200
    
    # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)
    
    # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20).to_frame(), 9).CLOSE)))) *
                (1 + rank(sma(self.returns, 25))))  # 250
    
    # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        return pow((self.high * self.low),0.5) - self.vwap
    
    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))
        
    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))
    
    # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) /5))) - rank((self.vwap - delay(self.vwap, 5))))
    
    # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
     
    
    # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha
    
    # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha050(self):
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))
    
    # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha
    
    # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 24) - ts_sum(self.returns, 4)) / 22))) * ts_rank(self.volume, 5))  # 240 20 220
        
    # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # This Alpha uses the Cap, however I have not acquired the data yet
    #    def alpha056(self):
    #        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))
        
    # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    def alpha057(self):
        return (0 - (1 * ((self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)).to_frame(), 2).CLOSE)))
    
    # Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
     
    # Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
     
    
    # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
    
	# Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha061(self):
        adv180 = sma(self.volume, 180)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18)))*1
    
	# Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha062(self):
        adv20 = sma(self.volume, 20)
        return ((rank(correlation(self.vwap, sma(adv20, 22), 10)) < rank(((rank(self.open) +rank(self.open)) < (rank(((self.high + self.low) / 2)) + rank(self.high))))) * -1)
    
    # Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
     
    
    # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha064(self):
        adv120 = sma(self.volume, 36)  # 120
        return ((rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),sma(adv120, 13), 17)) < rank(delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 -0.178404))), 3.69741))) * -1)
    
    # Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    def alpha065(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60,9), 6)) < rank((self.open - ts_min(self.open, 14)))) * -1)
      
    # Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha066(self):
        return ((rank(decay_linear(delta(self.vwap, 4).to_frame(), 7).CLOSE) + ts_rank(decay_linear(((((self.low* 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))).to_frame(), 11).CLOSE, 7)) * -1)
    
    # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
     
    
    # Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    def alpha068(self):
        adv15 = sma(self.volume, 15)
        return ((ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) <rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1.06157))) * -1)
    
    # Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)
         
    # Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
     
    
    # Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    def alpha071(self):
        adv180 = sma(self.volume, 180)
        p1=ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16)
        p2=ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'max']=df['p1']
        df.at[df['p2']>=df['p1'],'max']=df['p2']
        return df['max']
        #return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))
    
    # Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    def alpha072(self):
        adv40 = sma(self.volume, 40)
        return (rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9).to_frame(), 10).CLOSE) /rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7).to_frame(),3).CLOSE))
    
    # Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    def alpha073(self):
        p1=rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE)
        p2=ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'max']=df['p1']
        df.at[df['p2']>=df['p1'],'max']=df['p2']
        return -1*df['max']
        #return (max(rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE),ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)) * -1)
    
    # Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    def alpha074(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, sma(adv30, 37), 15)) <rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11)))* -1)*1
    
    # Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
    def alpha075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50),12)))*1
    
    # Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
     

    # Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    def alpha077(self):
        adv40 = sma(self.volume, 40)
        p1=rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE)
        p2=rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'min']=df['p2']
        df.at[df['p2']>=df['p1'],'min']=df['p1']
        return df['min']
        #return min(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE),rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE))
    
    # Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    def alpha078(self):
        adv40 = sma(self.volume, 40)
        # adv25 = sma(self.volume, 25)
        return (rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20),ts_sum(adv40,20), 7)).pow(rank(correlation(rank(self.vwap), rank(self.volume), 6))))
    
    # Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
     
    # Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
     
   
    # Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    def alpha081(self):
        adv10 = sma(self.volume, 10)
        return ((rank(log(product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50),8)).pow(4))), 15))) < rank(correlation(rank(self.vwap), rank(self.volume), 5))) * -1)
    
    # Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
     
    
    # Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    def alpha083(self):
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (((self.high -self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))
    
    # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    def alpha084(self):
        return pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close,5))
    
    # Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    def alpha085(self):
        adv30 = sma(self.volume, 30)
        return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30,10)).pow(rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10),7))))
    
    # Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)

    def alpha086(self):
        adv20 = sma(self.volume, 20)
        return ((ts_rank(correlation(self.close, sma(adv20, 15), 6), 20) < rank(((self.open+ self.close) - (self.vwap +self.open)))) * -1)
    
    # Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
     
    
    # Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    def alpha088(self):
        adv60 = sma(self.volume, 60)
        p1=rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),8).CLOSE)
        p2=ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,21), 8).to_frame(), 7).CLOSE, 3)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'min']=df['p2']
        df.at[df['p2']>=df['p1'],'min']=df['p1']
        return df['min']
        #return min(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),8).CLOSE), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,20.6966), 8).to_frame(), 7).CLOSE, 3))
    
    # Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
     
    # Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
     
    # Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
     

    # Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    def alpha092(self):
        adv30 = sma(self.volume, 30)
        p1=ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,19)
        p2=ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE,7)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'min']=df['p2']
        df.at[df['p2']>=df['p1'],'min']=df['p1']
        return df['min']
        #return  min(ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,19), ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE,7))
    
    # Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
     
    
    # Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    def alpha094(self):
        adv30 = sma(self.volume, 30)
        return ((rank((self.vwap - ts_min(self.vwap, 12))).pow(ts_rank(correlation(ts_rank(self.vwap,20), ts_rank(adv30, 4), 18), 3)) * -1))
    
    # Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    def alpha095(self):
        adv40 = sma(self.volume, 40)
        return (rank((self.open - ts_min(self.open, 12))) < ts_rank((rank(correlation(sma(((self.high + self.low)/ 2), 19), sma(adv40, 19), 13)).pow(5)), 12))*1
    
    # Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    def alpha096(self):
        adv60 = sma(self.volume, 60)
        p1=ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4),4).CLOSE, 8)
        p2=ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'max']=df['p1']
        df.at[df['p2']>=df['p1'],'max']=df['p2']
        return -1*df['max']
        #return (max(ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4),4).CLOSE, 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)) * -1)
    
    # Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
     
    
    # Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    def alpha098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        return (rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).CLOSE) -rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9),7).to_frame(), 8).CLOSE))
    
    # Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    def alpha099(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <rank(correlation(self.low, self.volume, 6))) * -1)
    
    # Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))
     

    # Alpha#101	 ((close - open) / ((high - low) + .001))
    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


class talib_index():

    def __init__(self, df):
        self.df = df.copy()
        self.open = df['open'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.close = df['close'].values
        self.volume = df['volume'].values

    def sma(self, n=None):
        """
        Simple moving average.
        """
        if n is None:
            return tb.SMA(self.close)
        else:
            return tb.SMA(self.close, n)

    def ema(self, n=None):
        """
        Exponential moving average.
        """
        if n is None:
            return tb.EMA(self.close)
        else:
            return tb.EMA(self.close, n)

    def kama(self, n=None):
        """
        KAMA.
        """
        if n is None:
            return tb.KAMA(self.close)
        else:
            return tb.KAMA(self.close, n)

    def wma(self, n=None):
        """
        WMA.
        """
        if n is None:
            return tb.WMA(self.close)
        else:
            return tb.WMA(self.close, n)

    def dema(self, n=None):
        """
        DEMA.
        """
        if n is None:
            return tb.DEMA(self.close)
        else:
            return tb.DEMA(self.close, n)

    def ht_trendline(self):
        """
        HT_TRENDLINE.
        """
        return tb.HT_TRENDLINE(self.close)
    
    def mama(self):
        """
        MAMA.
        """
        mama, fama =  tb.MAMA(self.close)
        return mama, fama
    
    def wavp(self, n=14):
        """
        WAVP.
        """
        return tb.MAVP(self.close, n)
    
    def minpoint(self, n=None):
        """
        MIDPOINT.
        """
        if n is None:
            return tb.MIDPOINT(self.close, timeperiod=14)
        else:
            return tb.MIDPOINT(self.close, timeperiod=n)

    def midprice(self, n=14):
        """
        MIDPRICE.
        """
        return tb.MIDPRICE(self.high, self.low, timeperiod=n)

    def sar(self):
        """
        SAR.
        """
        return tb.SAR(self.high, self.low, acceleration=0, maximum=0)

    def sarext(self, n=None):
        """
        SAREXT.
        """
        return tb.SAREXT(self.high, self.low, startvalue=0, offsetonreverse=0, 
                         accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, 
                         accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    def t3(self, n=5):
        """
        T3.
        """
        return tb.T3(self.close, timeperiod=n, vfactor=0)

    def trima(self, n=None):
        """
        TRIMA.
        """
        if n is None:
            return tb.TRIMA(self.close, timeperiod=30)
        else:
            return tb.TRIMA(self.close, timeperiod=n)

    def macdext(self):
        """
        MACDEXT .
        """
        macd, macdsignal, macdhist = tb.MACDEXT(self.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        return macd, macdsignal, macdhist

    def macdfix(self, n=None):
        """
        MACDFIX.
        """
        if n is None:
            macd, macdsignal, macdhist = tb.MACDFIX(self.close, signalperiod=9)
            return macd, macdsignal, macdhist
        else:
            macd, macdsignal, macdhist = tb.MACDFIX(self.close, signalperiod=n)
            return macd, macdsignal, macdhist

    def mfi(self, n=None):
        """
        MFI.
        """
        if n is None:
            return tb.MFI(self.high, self.low, self.close, self.volume, timeperiod=14)
        else:
            return tb.MFI(self.high, self.low, self.close, self.volume, timeperiod=n)

    def apo(self, fast_period=None, slow_period=None, matype=0):
        """
        APO.
        """
        if fast_period is None:
            return tb.APO(self.close)
        else:
            return tb.APO(self.close, fast_period, slow_period, matype)

    def cmo(self, n=None):
        """
        CMO.
        """
        if n is None:
            return tb.CMO(self.close)
        else:
            return tb.CMO(self.close, n)

    def mom(self, n=None):
        """
        MOM.
        """
        if n is None:
            return tb.MOM(self.close)
        else:
            return tb.MOM(self.close, n)

    def ppo(self, fast_period=None, slow_period=None, matype=0):
        """
        PPO.
        """
        if fast_period is None:
            return tb.PPO(self.close)
        else:
            return tb.PPO(self.close, fast_period, slow_period, matype)

    def roc(self, n=None):
        """
        ROC.
        """
        if n is None:
            return tb.MOM(self.close)
        else:
            return tb.MOM(self.close, n)

    def rocr(self, n=None):
        """
        ROCR.
        """
        if n is None:
            return tb.ROCR(self.close)
        else:
            return tb.ROCR(self.close, n)

    def rocp(self, n=None):
        """
        ROCP.
        """
        if n is None:
            return tb.ROCP(self.close)
        else:
            return tb.ROCP(self.close, n)

    def rocr_100(self, n=None):
        """
        ROCR100.
        """
        if n is None:
            return tb.ROCR100(self.close)
        else:
            return tb.ROCR100(self.close, n)

    def trix(self, n=10):
        """
        TRIX.
        """
        return tb.TRIX(self.close, n)

    def std(self, n=None, nbdev: int = 1):
        """
        Standard deviation.
        """
        if n is None:
            return tb.STDDEV(self.close)
        else:
            return tb.STDDEV(self.close, n, nbdev)

    def obv(self):
        """
        OBV.
        """
        return tb.OBV(self.close, self.volume)

    def cci(self, n=None):
        """
        Commodity Channel Index (CCI).
        """
        if n is None:
            return tb.CCI(self.high, self.low, self.close)
        else:
            return tb.CCI(self.high, self.low, self.close, n)

    def atr(self, n=None):
        """
        Average True Range (ATR).
        """
        if n is None:
            return tb.ATR(self.high, self.low, self.close)
        else:
            return tb.ATR(self.high, self.low, self.close, n)

    def natr(self, n=None):
        """
        NATR.
        """
        if n is None:
            return tb.NATR(self.high, self.low, self.close)
        else:
            return tb.NATR(self.high, self.low, self.close, n)

    def rsi(self, n=None):
        """
        Relative Strenght Index (RSI).
        """
        if n is None:
            return tb.RSI(self.close)
        else:
            return tb.RSI(self.close, n)

    def stoch(self):
        """
        STOCH.
        """
        slowk, slowd = tb.STOCH(self.high, self.low, self.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        return slowk, slowd 

    def stochf(self):
        """
        STOCHF.
        """
        fastk, fastd = tb.STOCHF(self.high, self.low, self.close, fastk_period=5, fastd_period=3, fastd_matype=0)
        return fastk, fastd

    def stochrsi(self, n=None):
        """
        STOCHRSI.
        """
        if n is None:
            fastk, fastd = tb.STOCHRSI(self.close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
            return fastk, fastd
        else:
            fastk, fastd = tb.STOCHRSI(self.close, timeperiod=n, fastk_period=5, fastd_period=3, fastd_matype=0)
            return fastk, fastd

    def macd(self, fastperiod=12, slowperiod=26, signalperiod=9):
        """
        MACD.
        """
        macd, macdsignal, macdhist = tb.MACD(self.close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return macd, macdsignal, macdhist
            
    def adx(self, n=None):
        """
        ADX.
        """
        if n is None:
            return tb.ADX(self.high, self.low, self.close)
        else:
            return tb.ADX(self.high, self.low, self.close, n)

    def adxr(self, n=None):
        """
        ADXR.
        """
        if n is None:
            return tb.ADXR(self.high, self.low, self.close)
        else:
            return tb.ADXR(self.high, self.low, self.close, n)

    def dx(self, n=None):
        """
        DX.
        """
        if n is None:
            return tb.DX(self.high, self.low, self.close)
        else:
            return tb.DX(self.high, self.low, self.close, n)

    def minus_di(self, n=None):
        """
        MINUS_DI.
        """
        if n is None:
            return tb.MINUS_DI(self.high, self.low, self.close)
        else:
            return tb.MINUS_DI(self.high, self.low, self.close, n)

    def plus_di(self, n=None):
        """
        PLUS_DI.
        """
        if n is None:
            return tb.PLUS_DI(self.high, self.low, self.close)
        else:
            return tb.PLUS_DI(self.high, self.low, self.close, n)

    def willr(self, n=None):
        """
        WILLR.
        """
        if n is None:
            return tb.WILLR(self.high, self.low, self.close)
        else:
            return tb.WILLR(self.high, self.low, self.close, n)

    def ultosc(self, time_period1: int = 7, time_period2: int = 14, time_period3: int = 28):
        """
        Ultimate Oscillator.
        """
        return tb.ULTOSC(self.high, self.low, self.close, time_period1, time_period2, time_period3)

    def trange(self):
        """
        TRANGE.
        """
        return tb.TRANGE(self.high, self.low, self.close)

    def boll(self, n=None, dev=1.5):
        """
        Bollinger Channel.
        """
        if n is None:
            mid = self.sma()
            std = self.std()
            up = mid + std * dev
            down = mid - std * dev
            return up, down
        else:
            mid = self.sma(n)
            std = self.std(n)
            up = mid + std * dev
            down = mid - std * dev
            return up, down
        
    def keltner(self, n=None, dev=1.5):
        """
        Keltner Channel.
        """
        if n is None:
            mid = self.sma()
            atr = self.atr()
        else:
            mid = self.sma(n)
            atr = self.atr(n)

        up = mid + atr * dev
        down = mid - atr * dev

        return up, down

    def avgprice(self):
        """
        AVGPRICE.
        """
        return tb.AVGPRICE(self.open, self.high, self.low, self.close)

    def medprice(self):
        """
        MEDPRICE.
        """
        return tb.MEDPRICE(self.high, self.low)

    def typprice(self):
        """
        TYPPRICE.
        """
        return tb.TYPPRICE(self.high, self.low, self.close)
    
    def wclprice(self):
        """
        WCLPRICE.
        """
        return tb.WCLPRICE(self.high, self.low, self.close)

    def add(self):
        """
        ADD
        """
        return tb.ADD(self.high, self.low)

    def div(self):
        """
        DIV
        """
        return tb.DIV(self.high, self.low)

    def max(self, n=None):
        """
        MAX.
        """
        if n is None:
            return tb.MAX(self.close, timeperiod=30)
        else:
            return tb.MAX(self.close, timeperiod=n)

    def maxindex(self, n=None):
        """
        MAXINDEX.
        """
        if n is None:
            return tb.MAXINDEX(self.close, timeperiod=30)
        else:
            return tb.MAXINDEX(self.close, timeperiod=n)
        
    def min(self, n=None):
        """
        MIN.
        """
        if n is None:
            return tb.MIN(self.close, timeperiod=30)
        else:
            return tb.MIN(self.close, timeperiod=n)
        
    def minindex(self, n=None):
        """
        minindex.
        """
        if n is None:
            return tb.MININDEX(self.close, timeperiod=30)
        else:
            return tb.MININDEX(self.close, timeperiod=n)
        
    def minmax(self, n=None):
        """
        MINMAX.
        """
        if n is None:
            min, max = tb.MINMAX(self.close, timeperiod=30)
            return min, max
        else:
            min, max = tb.MINMAX(self.close, timeperiod=n)
            return min, max

    def minmaxindex(self, n=None):
        """
        MINMAXINDEX.
        """
        if n is None:
            minidx, maxidx = tb.MINMAXINDEX(tb.close, timeperiod=30)
            return minidx, maxidx
        else:
            minidx, maxidx = tb.MINMAXINDEX(tb.close, timeperiod=n)
            return minidx, maxidx

    def mult(self):
        """
        MULT.
        """
        return tb.MULT(self.high, self.low)

    def sub(self):
        """
        SUB.
        """
        return tb.SUB(self.high, self.low)

    def sum(self, n=30):
        return tb.SUM(self.close, timeperiod=n)
    
    def ht_dcperiod(self):
        return tb.HT_DCPERIOD(self.close)

    def ht_dcphase(self):
        return tb.HT_DCPHASE(self.close)

    def ht_phasor(self):
        inphase, quadrature = tb.HT_PHASOR(self.close)
        return inphase, quadrature 

    def ht_sine(self):
        sine, leadsine = tb.HT_SINE(self.close)
        return sine, leadsine
    
    def ht_trendmode(self):
        return tb.HT_TRENDMODE(self.close)

    def cdl2crows(self):
        """两只乌鸦"""
        return tb.CDL2CROWS(self.open, self.high, self.low, self.close)
    
    def cdl3blackcrows(self):
        """三只乌鸦"""
        return tb.CDL3BLACKCROWS(self.open, self.high, self.low, self.close)
    
    def cdl3inside(self):
        """三内部上涨和下跌"""
        return tb.CDL3INSIDE(self.open, self.high, self.low, self.close)
    
    def cdl3instrike(self):
        """三线打击"""
        return tb.CDL3LINESTRIKE(self.open, self.high, self.low, self.close)
    
    def cdl3outside(self):
        """三外部上涨和下跌"""
        return tb.CDL3OUTSIDE(self.open, self.high, self.low, self.close)

    def cdl3starsinsouth(self):
        """南方三星"""
        return tb.CDL3STARSINSOUTH(self.open, self.high, self.low, self.close)

    def cdl3whitsoldiers(self):
        """三个白兵"""
        return tb.CDL3WHITESOLDIERS(self.open, self.high, self.low, self.close)
    
    def cdabandonbaby(self):
        """弃婴"""
        return tb.CDLABANDONEDBABY(self.open, self.high, self.low, self.close)

    def cdbelthold(self):
        """捉腰带线"""
        return tb.CDLBELTHOLD(self.open, self.high, self.low, self.close)

    def cdadvanceblock(self):
        """大敌当前"""
        return tb.CDLADVANCEBLOCK(self.open, self.high, self.low, self.close)

    def cdbreakaway(self):
        """脱离"""
        return tb.CDLBREAKAWAY(self.open, self.high, self.low, self.close)

    def cdclosingmarubozu(self):
        """收盘缺影线"""
        return tb.CDLCLOSINGMARUBOZU(self.open, self.high, self.low, self.close)

    def cdconcealingbabyswallow(self):
        """藏婴吞没"""
        return tb.CDLCONCEALBABYSWALL(self.open, self.high, self.low, self.close)

    def cddarkcloudcover(self):
        """乌云压顶"""
        return tb.CDLCOUNTERATTACK(self.open, self.high, self.low, self.close)
    
    def cdldoji(self):
        """Doji 十字"""
        return tb.CDLDOJI(self.open, self.high, self.low, self.close)

    def cdldojistar(self):
        """Doji Star十字星"""
        return tb.CDLDOJISTAR(self.open, self.high, self.low, self.close)

    def cdldragonflydoji(self):
        """T形十字"""
        return tb.CDLDRAGONFLYDOJI(self.open, self.high, self.low, self.close)

    def cdlengulfing(self):
        """吞噬模式"""
        return tb.CDLENGULFING(self.open, self.high, self.low, self.close)

    def cdconcealingbabyswallow(self):
        """藏婴吞没"""
        return tb.CDLCONCEALBABYSWALL(self.open, self.high, self.low, self.close)

    def cdleveningdojistar(self):
        """十字暮星"""
        return tb.CDLEVENINGDOJISTAR(self.open, self.high, self.low, self.close)
    
    def cdleveingstar(self):
        """暮星"""
        return tb.CDLEVENINGSTAR(self.open, self.high, self.low, self.close)

    def cdldojistar(self):
        """Doji Star十字星"""
        return tb.CDLDOJISTAR(self.open, self.high, self.low, self.close)

    def cdlgapsidewhite(self):
        """向上/下跳空并列阳线"""
        return tb.CDLGAPSIDESIDEWHITE(self.open, self.high, self.low, self.close)

    def cdlggravestonedoji(self):
        """ 墓碑十字/倒T十字"""
        return tb.CDLGRAVESTONEDOJI(self.open, self.high, self.low, self.close)

    def cdlhammer(self):
        """ 锤头"""
        return tb.CDLHAMMER(self.open, self.high, self.low, self.close)

    def cdlhangingman(self):
        """上吊线"""
        return tb.CDLHANGINGMAN(self.open, self.high, self.low, self.close)
    
    def cdlharami(self):
        """母子线"""
        return tb.CDLHARAMI(self.open, self.high, self.low, self.close)

    def cdlharamicross(self):
        """十字孕线"""
        return tb.CDLHARAMICROSS(self.open, self.high, self.low, self.close)

    def cdlhighwave(self):
        """风高浪大线"""
        return tb.CDLHIGHWAVE(self.open, self.high, self.low, self.close)

    def cdlhikkake(self):
        """ 陷阱"""
        return tb.CDLHIKKAKE(self.open, self.high, self.low, self.close)

    def cdlhikkakemode(self):
        """修正陷阱"""
        return tb.CDLHIKKAKEMOD(self.open, self.high, self.low, self.close)
    
    def cdlhomingpigeon(self):
        """家鸽"""
        return tb.CDLHOMINGPIGEON(self.open, self.high, self.low, self.close)

    def cdlidentical3crows(self):
        """三胞胎乌鸦"""
        return tb.CDLIDENTICAL3CROWS(self.open, self.high, self.low, self.close)

    def sdlinneck(self):
        """颈内线"""
        return tb.CDLINNECK(self.open, self.high, self.low, self.close)

    def cdlinvertedhammer(self):
        """ 倒锤头"""
        return tb.CDLINVERTEDHAMMER(self.open, self.high, self.low, self.close)

    def cdlkicking(self):
        """反冲形态"""
        return tb.CDLKICKING(self.open, self.high, self.low, self.close)

    def cdlkickingbylength(self):
        """由较长缺影线决定的反冲形态"""
        return tb.CDLKICKINGBYLENGTH(self.open, self.high, self.low, self.close)

    def cdlladderbottom(self):
        """梯底"""
        return tb.CDLLADDERBOTTOM(self.open, self.high, self.low, self.close)
    
    def cdllongleggeddoji(self):
        """长脚十字"""
        return tb.CDLLONGLEGGEDDOJI(self.open, self.high, self.low, self.close)

    def cdllongline(self):
        """长蜡烛"""
        return tb.CDLLONGLINE(self.open, self.high, self.low, self.close)

    def cdlmarubozu(self):
        """光头光脚/缺影线"""
        return tb.CDLMARUBOZU(self.open, self.high, self.low, self.close)

    def cdlmatchhighlow(self):
        """ 相同低价"""
        return tb.CDLMATCHINGLOW(self.open, self.high, self.low, self.close)
    
    def cdlmathold(self):
        """ 铺垫"""
        return tb.CDLMATHOLD(self.open, self.high, self.low, self.close)

    def cdlmorningdojistar(self):
        """十字晨星"""
        return tb.CDLMORNINGDOJISTAR(self.open, self.high, self.low, self.close)

    def cdlmorningstar(self):
        """晨星"""
        return tb.CDLMORNINGSTAR(self.open, self.high, self.low, self.close)

    def cdlonneck(self):
        """颈上线"""
        return tb.CDLONNECK(self.open, self.high, self.low, self.close)
    
    def cdlpiercing(self):
        """刺透形态"""
        return tb.CDLPIERCING(self.open, self.high, self.low, self.close)

    def cdlrickshawman(self):
        """黄包车夫"""
        return tb.CDLRICKSHAWMAN(self.open, self.high, self.low, self.close)

    def cdlrisefall3methods(self):
        """上升/下降三法"""
        return tb.CDLRISEFALL3METHODS(self.open, self.high, self.low, self.close)

    def cdlseparatinglines(self):
        """ 分离线"""
        return tb.CDLSEPARATINGLINES(self.open, self.high, self.low, self.close)
    
    def cdlshootingstar(self):
        """射击之星"""
        return tb.CDLSHOOTINGSTAR(self.open, self.high, self.low, self.close)

    def cdlshortline(self):
        """短蜡烛"""
        return tb.CDLSHORTLINE(self.open, self.high, self.low, self.close)

    def cdlspinningtop(self):
        """纺锤"""
        return tb.CDLSPINNINGTOP(self.open, self.high, self.low, self.close)
    
    def cdlstalledpattern(self):
        """停顿形态"""
        return tb.CDLSTALLEDPATTERN(self.open, self.high, self.low, self.close)

    def cdlsticksandwich(self):
        """条形三明治"""
        return tb.CDLSTICKSANDWICH(self.open, self.high, self.low, self.close)

    def cdltankurl(self):
        """探水竿"""
        return tb.CDLTAKURI(self.open, self.high, self.low, self.close)

    def cdltasukigap(self):
        """ 跳空并列阴阳线"""
        return tb.CDLTASUKIGAP(self.open, self.high, self.low, self.close)

    def cdlthrusting(self):
        """插入"""
        return tb.CDLTHRUSTING(self.open, self.high, self.low, self.close)

    def cdltristar(self):
        """三星"""
        return tb.CDLTRISTAR(self.open, self.high, self.low, self.close)
    
    def cdlunique3river(self):
        """奇特三河床"""
        return tb.CDLUNIQUE3RIVER(self.open, self.high, self.low, self.close)

    def cdlupsidegap2crows(self):
        """向上跳空的两只乌鸦"""
        return tb.CDLUPSIDEGAP2CROWS(self.open, self.high, self.low, self.close)

    def cdlxssidegap3methods(self):
        """ 上升/下降跳空三法"""
        return tb.CDLXSIDEGAP3METHODS(self.open, self.high, self.low, self.close)

    def beta(self, n=5):
        """贝塔系数"""
        return tb.BETA(self.high, self.low, timeperiod=n)

    def correl(self, n=30):
        """皮尔逊相关系数"""
        return tb.CORREL(self.high, self.low, timeperiod=n)

    def linearreg(self, n=14):
        """线性回归"""
        return tb.LINEARREG(self.close, timeperiod=n)

    def linerreg_angle(self, n=14):
        """线性回归的角度"""
        return tb.LINEARREG_ANGLE(self.close, timeperiod=n)

    def linearreg_intercept(self, n=14):
        """线性回归截距"""
        return tb.LINEARREG_INTERCEPT(self.close, timeperiod=n)

    def linearreg_slope(self, n=14):
        """线性回归斜率指标"""
        return tb.LINEARREG_SLOPE(self.close, timeperiod=n)

    def tsf(self, n=14):
        """时间序列预测"""
        return tb.TSF(self.close, timeperiod=n)
    
    def var(self, n=5):
        """方差"""
        return tb.VAR(self.close, timeperiod=n)

    # def acos(self):
    #     """反余弦函数"""
    #     return tb.ACOS(self.close)

    # def asin(self):
    #     """反正弦函数"""
    #     return tb.ASIN(self.close)

    # def atan(self):
    #     """反正切值"""
    #     return tb.ATAN(self.close)

    # def ceil(self):
    #     """向上取整数"""
    #     return tb.CEIL(self.close)

    # def cos(self):
    #     """余弦函数"""
    #     return tb.COS(self.close)

    # def cosh(self):
    #     """双曲正弦函数"""
    #     return tb.COSH(self.close)

    # def exp(self):
    #     """指数曲线"""
    #     return tb.EXP(self.close)

    # def floor(self):
    #     """向下取整数"""
    #     return tb.FLOOR(self.close)

    # def ln(self):
    #     """自然对数"""
    #     return tb.LN(self.close)

    # def log10(self):
    #     """对数函数log"""
    #     return tb.LOG10(self.close)

    # def sin(self):
    #     """正弦函数"""
    #     return tb.SIN(self.close)

    # def sinh(self):
    #     """双曲正弦函数"""
    #     return tb.SINH(self.close)

    # def sqrt(self):
    #     """非负实数的平方根"""
    #     return tb.SQRT(self.close)

    # def tan(self):
    #     """对数函数log"""
    #     return tb.TAN(self.close)

    def aroon(self, n=None):
        """
        Aroon indicator.
        """
        if n is None:
            aroon_down, aroon_up = tb.AROON(self.high, self.low)
            return aroon_down, aroon_up
        else:
            aroon_down, aroon_up = tb.AROON(self.high, self.low, n)
            return aroon_down, aroon_up

    def aroonosc(self, n=None):
        """
        Aroon Oscillator.
        """
        if n is None:
            return tb.AROONOSC(self.high, self.low)
        else:
            return tb.AROONOSC(self.high, self.low, n)

    def ad(self):
        """
        AD.
        """
        return tb.AD(self.high, self.low, self.close, self.volume)

    def adosc(self):
        """
        ADOSC.
        """
        return tb.ADOSC(self.high, self.low, self.close, self.volume, fastperiod=3, slowperiod=10)

    def bop(self):
        """
        BOP.
        """
        return tb.BOP(self.open, self.high, self.low, self.close)

    def minus_dm(self, n=5):
        """
        MINUS_DM.
        """
        result = tb.MINUS_DM(self.high, self.low, n)
        return result

    def plus_dm(self, n=10):
        """
        PLUS_DM.
        """
        result = tb.PLUS_DM(self.high, self.low, n)

        return result


class AdjIndex(talib_index):
    '''把非平稳序列改成类平稳序列'''
    def __init__(self, df):
        super().__init__(df)
    
    def atrzc(self, n=None):
        """
        Average True Range (ATR).
        """
        if n is None:
            pre_close = np.array([np.nan]*len(self.close))
            pre_close[6:] = self.close[:-6]
            return np.sign(self.close-pre_close)*tb.ATR(self.high, self.low, self.close)
        else:
            pre_close = np.array([np.nan]*len(self.close))
            pre_close[n:] = self.close[:-n]
            return np.sign(self.close-pre_close)*tb.ATR(self.high, self.low, self.close, n)

    def natrzc(self, n=None):
        """
        NATR.
        """
        if n is None:
            pre_close = np.array([np.nan]*len(self.close))
            pre_close[6:] = self.close[:-6]
            return np.sign(self.close-pre_close)*tb.NATR(self.high, self.low, self.close)
        else:
            pre_close = np.array([np.nan]*len(self.close))
            pre_close[n:] = self.close[:-n]
            return np.sign(self.close-pre_close)*tb.NATR(self.high, self.low, self.close, n)
    
    def trangezc(self):
        """
        TRANGE.
        """
        pre_close = np.array([np.nan]*len(self.close))
        pre_close[1:] = self.close[:-1]
        return np.sign(self.close-pre_close)*tb.TRANGE(self.high, self.low, self.close)

    def stdzc(self, n=None, nbdev: int = 1):
        """
        Standard deviation.
        """
        if n is None: n = 6
        pre_close = np.array([np.nan]*len(self.close))
        pre_close[n:] = self.close[:-n]
        return np.sign(self.close-pre_close)*tb.STDDEV(self.close, n, nbdev)

    def subzc(self):
        """
        SUB.
        """
        pre_close = np.array([np.nan]*len(self.close))
        pre_close[1:] = self.close[:-1]
        return np.sign(self.close-pre_close)*tb.SUB(self.high, self.low)

    def varzc(self, n=5):
        """方差"""
        pre_close = np.array([np.nan]*len(self.close))
        pre_close[1:] = self.close[:-1]
        return np.sign(self.close-pre_close)*tb.VAR(self.close, timeperiod=n)

    # def minus_dmzc(self, n=5):
    #     """
    #     MINUS_DM.
    #     """
    #     result = np.log(1+tb.MINUS_DM(self.high, self.low, n))
    #     return result

    # def plus_dmzc(self, n=10):
    #     """
    #     PLUS_DM.
    #     """
    #     result = np.log(1+tb.PLUS_DM(self.high, self.low, n))
    #     return result

    # def minus_dizc(self, n=None):
    #     """
    #     MINUS_DI.
    #     """
    #     if n is None:
    #         return np.log(1+tb.MINUS_DI(self.high, self.low, self.close))
    #     else:
    #         return np.log(1+tb.MINUS_DI(self.high, self.low, self.close, n))

    # def plus_dizc(self, n=None):
    #     """
    #     PLUS_DI.
    #     """
    #     if n is None:
    #         return np.log(1+tb.PLUS_DI(self.high, self.low, self.close))
    #     else:
    #         return np.log(1+tb.PLUS_DI(self.high, self.low, self.close, n))

    def obvzc(self):
        """
        OBV.
        """
        pre_volume = np.array([np.nan]*len(self.volume))
        pre_volume[1:] = self.volume[:-1]
        return tb.OBV(self.close, self.volume) / pre_volume

class GPLearnIndex():

    def __init__(self):
        self.kline_col = ['open', 'high', 'low', 'close', 'volume', 'turnover']

    def set_datas(self, df):
        self.df = df.copy()
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']
        self.turnover = df['turnover']
    
    def get_index(self, index_name: str, method='', params=None, just_index_v=0):
        '''将因子名变成可执行函数并输出因子值'''
        index_name_li = index_name.split('__')
        params_li = [eval(index_name_li[0])] if '_' not in index_name_li[0] else index_name_li[0].split('_')
        if len(method) == 0: method = f'_{index_name_li[1]}'
        index_name = index_name_li[-1]
        # li = index_name.split('(')
        li1 = set(re.findall('[a-zA-Z_]+', index_name))
        for i in li1:
            if i in self.kline_col:
                index_name = index_name.replace(i, f'self.{i}')
            else:
                index_name = index_name.replace(i, f'self._{i}')

        index_name = index_name.replace(f'self._self._self._', f'self._')
        index_name = index_name.replace(f'self._self._', f'self._')
        index_s = eval(index_name)

        # index_v = index_s.iloc[-1]
        index_v = index_s.copy()

        if just_index_v:
            return index_v

        if len(params_li) > 1:
            index_s = getattr(self, method)(index_s, *[eval(i) for i in params_li[1:]])
        else:
            index_s = getattr(self, method)(index_s)
        # print(index_s.iloc[-1], index_v, [eval(i) for i in params_li[1:]])
        return index_s, index_v

    def _df_apply(self, x, func, window):
        res_s = x.rolling(window).apply(lambda x: func(x))
        return res_s
    
    def _shape_arr(self, x, shape):
        '''转换矩阵shape，多余部分用nan填充'''
        arr = np.array([np.nan]*shape)
        if len(x): arr[-x.shape[0]:] = x
        return arr

    def _cov_np(self, x1, x2):
        left = x1
        right = x2
        
        # 按前值填充缺失值
        # left = fill_nan(left)
        # right = fill_nan(right)
        
        # 协方差计算
        ldem = left - np.nanmean(left, axis = 0)
        rdem = right - np.nanmean(right, axis = 0)
        num = ldem * rdem
        col_sum = np.nansum(num, axis = 0)
        # col_sum[col_sum == 0] = np.nan
        cov = col_sum / x1.shape[0]
        
        return cov

    def _apply_x1(self, x1, func):
        df = pd.DataFrame({'x1': x1.values})
        index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        # res = func(df_adj)
        res = pd.Series(self._shape_arr(func(df_adj), len(x1)))
        return res
    
    def _apply_x1x2(self, x1, x2, func):
        # print(x1)
        # print(x2)
        
        df = pd.DataFrame({'x1': x1.values, 'x2': x2.values})
        index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        # res = func(df_adj)
        res = pd.Series(self._shape_arr(func(df_adj), len(x1)))
        return res

    def _apply_xn(self, x_li, func):
        x_dic = {}
        for j in range(len(x_li)):
            x_dic[f'x{j}'] = x_li[j].values
        df = pd.DataFrame(x_dic)
        df_adj = df.dropna()
        index_i = df_adj.index
        # res = func(df_adj)
        res = pd.Series(self._shape_arr(func(*[df_adj.iloc[:, i].values for i in range(len(x_li))]), len(df)))
        return res

    def _apply_xn_d(self, x_li, func, window, window1=None):
        if window1 is None:
            window1 = window
        x_dic = {}
        for j in range(len(x_li)):
            x_dic[f'x{j}'] = x_li[j].values
        df = pd.DataFrame(x_dic)
        df_adj = df.dropna()
        if len(df_adj) < window: return pd.Series([np.nan]*len(df))
        index_i = df_adj.index
        res = [func(*[df_adj.iloc[k-window+1:k+1].values[:,l] for l in range(df_adj.shape[1])], window1)[-1] if k >= window -1
                else np.nan for k in range(len(df_adj))]
        res = pd.Series(self._shape_arr(np.array(res), len(df)))
        return res

    def _ts_rank(self, x, window):
        '''当所在前window个分位数'''
        def rank_quantile(x:pd.Series):
            return x.rank().iloc[-1] / len(x)
        return self._df_apply(x, rank_quantile, window)

    def _ts_delay(self, x, window):
        '''window天之前的值'''
        def delay(x):
            return x.iloc[0]
        return self._df_apply(x, delay, window)

    def _ts_correlation(self, x1, x2, window):
        '''x1, x2前window个相关系数'''
        df = pd.DataFrame({'x1': x1, 'x2': x2})
        df_adj = df.dropna()
        index_i = df_adj.index
        res = [df_adj.iloc[j-window:j].corr().iloc[0, 1] if j >= window else np.nan for j in range(len(df_adj))]
        res = pd.Series(self._shape_arr(np.array(res), len(df)))
        return res

    def _ts_covariance(self, x1, x2, window):
        '''x1, x2前window个相关系数'''
        df = pd.DataFrame({'x1': x1, 'x2': x2})
        index_i = df[~df['x1'].isnull()].index
        df_adj = df.dropna()
        res = [self._cov_np(df_adj.iloc[j-window:j, 0].values, df_adj.iloc[j-window:j, 1].values) if j >= window else np.nan for j in range(len(df_adj))]
        res = pd.Series(self._shape_arr(np.array(res), len(index_i)))
        return res

    def _ts_scale(self, x, window):
        '''求当前值相对前window个值的百分比'''
        def scale(x):
            return x.iloc[-1] / len(x)
        return self._df_apply(x, scale, window)

    def _ts_delta(self, x, window):
        '''x和前window的x的差值'''
        def delta(x):
            return x.iloc[-1] - x.iloc[0]
        return self._df_apply(x, delta, window)

    def _ts_decay_linear(self, x, window):
        '''前window个加权平均'''
        def decay_linear(x):
            return np.mean(x.values*np.arange(1, len(x)+1))
        return self._df_apply(x, decay_linear, window)

    def _ts_min(self, x, window):
        '''前win个最小值'''
        return self._df_apply(x, np.min, window)

    def _ts_max(self, x, window):
        '''前win个最大值'''
        return self._df_apply(x, np.max, window)

    def _ts_argmax(self, x, window):
        '''前win个最大值的索引'''
        return self._df_apply(x, np.argmax, window)

    def _ts_argmin(self, x, window):
        '''前win个最小值索引'''
        return self._df_apply(x, np.argmin, window)

    def _ts_sum(self, x, window):
        '''前win个值之和'''
        return self._df_apply(x, np.sum, window)

    def _ts_product(self, x, window):
        '''前win个值之积'''
        return self._df_apply(x, np.prod, window)

    def _ts_std(self, x, window):
        '''前win个值标准差'''
        return self._df_apply(x, np.std, window)

    def _ts_zscore(self, x, window):
        '''(X-mean(X, window))/std(X, window)'''
        def zscore(x):
            return (x.iloc[-1]-np.mean(x))/np.std(x)
        return self._df_apply(x, zscore, window)

    def _ts_skewness(self, x, window):
        '''前win个峰度'''
        def skew(x):
            return x.skew()
        return self._df_apply(x, skew, window)

    def _ts_kurtosis(self, x, window):
        '''前win个峰度'''
        def kurt(x):
            return x.kurt()
        return self._df_apply(x, kurt, window)

    def _ts_max_diff(self, x, window):
        '''最大值和x差值'''
        def max_diff(x):
            return x.iloc[-1]-np.max(x)
        return self._df_apply(x, max_diff, window)

    def _ts_min_diff(self, x, window):
        '''最小值和x差值'''
        def min_diff(x):
            return x.iloc[-1]-np.min(x)
        return self._df_apply(x, min_diff, window)

    def _ts_return(self, x, window):
        '''前win的return'''
        def m_return(x):
            return 0 if x.iloc[0] == 0 else (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        return self._df_apply(x, m_return, window)

    def _ts_sharp(self, x, window):
        '''前win的ir'''
        def sharp(x):
            return 0 if np.std(x) == 0 else np.mean(x) / np.std(x)
        return self._df_apply(x, sharp, window)

    def _ts_median(self, x, window):
        '''前win的中位数'''
        return self._df_apply(x, np.median, window)

    def _ts_zscore_square(self, x, window):
        def zscore_square(x):
            return 0 if np.std(x) == 0 else ((x.iloc[-1] - np.mean(x)) / np.std(x))**2
        return self._df_apply(x, zscore_square, window)

    def _sigmoid(self, x):
        '''sigmoid X'''
        def sigmoid(df_adj):
            x = df_adj['x1'].values
            return 1/(1+np.exp(-x))
        return self._apply_x1(x, sigmoid)

    def _log(self, x):
        '''log x'''
        def m_log(df_adj):
            x = df_adj['x1'].values
            return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.)
        return self._apply_x1(x, m_log)

    def _abs(self, x):
        '''abs x'''
        def m_abs(df_adj):
            return np.abs(df_adj['x1'].values)
        return self._apply_x1(x, m_abs)

    def _sqrt(self, x):
        '''sqrt x'''
        def m_sqrt(df_adj):
            x = df_adj['x1'].values
            return np.sqrt(np.abs(x))
        return self._apply_x1(x, m_sqrt)

    def _sin(self, x):
        '''sin x'''
        def m_sin(df_adj):
            x = df_adj['x1'].values
            return np.sin(x)
        return self._apply_x1(x, m_sin)

    def _cos(self, x):
        '''sin x'''
        def m_cos(df_adj):
            x = df_adj['x1'].values
            return np.cos(x)
        return self._apply_x1(x, m_cos)

    def _sign(self, x):
        '''sin x'''
        def m_sign(df_adj):
            x = df_adj['x1'].values
            return np.sign(x)
        return self._apply_x1(x, m_sign)

    def _add(self, x1, x2):
        '''add'''
        def m_add(df_adj):
            return np.add(df_adj['x1'].values, df_adj['x2'].values)
        return self._apply_x1x2(x1, x2, m_add)

    def _reduce(self, x1, x2):
        ''''''
        def m_reduce(df_adj):
            return df_adj['x1'].values - df_adj['x2'].values
        return self._apply_x1x2(x1, x2, m_reduce)

    def _multiply(self, x1, x2):
        ''''''
        def m_multiply(df_adj):
            return df_adj['x1'].values * df_adj['x2'].values
        return self._apply_x1x2(x1, x2, m_multiply)

    def _division(self, x1, x2):
        """Closure of division (x1/x2) for zero denominator."""
        def m_division(df_adj):
            x1, x2 = df_adj['x1'].values, df_adj['x2'].values
            return x1/x2 # np.divide(x1, x2) # np.where(np.abs(x2) > 0.00001, np.divide(x1, x2), 1.)
        return self._apply_x1x2(x1, x2, m_division)

    def _shift(self, x):
        return pd.Series(x).shift(1)

    # 2022.9.7 new add
    def _ts_mean_return(self, x, window):
        '''window个x的变化率的'''
        def mean_return(x: pd.Series):
            return x.pct_change().mean()
        return self._df_apply(x, mean_return, window)

    # ta-lib的函数
    def _ts_dema(self, x, window):
        ''''''
        def dema(x):
            return tb.DEMA(x.values, (len(x)+1)/2)[-1]
        return self._df_apply(x, dema, window)

    def _ts_kama(self, x, window):
        ''''''
        def kama(x):
            return tb.KAMA(x.values, len(x)-1)[-1]   
        return self._df_apply(x, kama, window)    

    def _ts_ma(self, x, window):
        def ma(x):
            return tb.MA(x.values, len(x))[-1]
        return self._df_apply(x, ma, window)

    def _ts_midpoint(self, x, window):
        '''win前最大值和最小值的均值'''
        def midpoint(x):
            return tb.MIDPOINT(x.values, len(x))
        return self._df_apply(x, midpoint, window)

    def _ts_midprice(self, x1, x2, window):
        '''mean(max(x1,win), min(x2,win))'''
        return self._apply_xn_d([x1, x2], tb.MIDPRICE, window)

    def _ts_aroonosc(self, x1, x2, window):
        return self._apply_xn_d([x1, x2], tb.AROONOSC, window, window-1)

    def _ts_willr(self, x1, x2, x3, window):
        return self._apply_xn_d([x1, x2, x3], tb.WILLR, window)

    def _ts_cci(self, x1, x2, x3, window):
        return self._apply_xn_d([x1, x2, x3], tb.CCI, window)

    def _ts_adx(self, x1, x2, x3, window):
        return self._apply_xn_d([x1, x2, x3], tb.ADX, window)

    def _ts_mfi(self, x1, x2, x3, x4, window):
        return self._apply_xn_d([x1, x2, x3, x4], tb.MFI, window)

    def _ts_natr(self, x1, x2, x3, window):
        return self._apply_xn_d([x1, x2, x3], tb.NATR, window)

    def _ts_beta(self, x1, x2, window):
        def beta(x1, x2, window):
            x1_return = (x1[1:]-x1[:-1]) / x1[:-1]
            x2_return = (x2[1:]-x2[:-1]) / x2[:-1]
            res = st.linregress(x1_return, x2_return)[0]
            return [res]
        return self._apply_xn_d([x1, x2], beta, window)

    def _ts_linearreg_angle(self, x, window):
        return self._apply_xn_d([x], tb.LINEARREG_ANGLE, window)

    def _ts_linearreg_intercept(self, x, window):
        return self._apply_xn_d([x], tb.LINEARREG_INTERCEPT, window)

    def _ts_linearreg_slope(self, x, window):
        return self._apply_xn_d([x], tb.LINEARREG_SLOPE, window)

    def _ht_dcphase(self, x):
        def ht_dcphase(df_adj):
            x = df_adj['x1'].values
            return tb.HT_DCPHASE(x)
        return self._apply_x1(x, ht_dcphase)

    def _neg(self, x):
        '''相反数'''
        def neg(df_adj):
            x = df_adj['x1'].values
            return -x
        return self._apply_x1(x, neg)
    
    # 2022.9.13 add
    
    def _ts_wma(self, x, window):
        """
        WMA.
        """
        def wma(x):
            return tb.WMA(x.values, len(x))[-1]
        return self._df_apply(x, wma, window)

    def _ts_trima(self, x, window):
        """
        TRIMA.
        """
        def func(x):
            return tb.TRIMA(x.values, len(x))[-1]
        return self._df_apply(x, func, window)

    def _ts_cmo(self, x, window):
        """
        CMO.
        """
        return self._apply_xn_d([x], tb.CMO, window, window-1)

    def _ts_mom(self, x, window):
        """
        MOM.
        """
        return self._apply_xn_d([x], tb.MOM, window, window-1)

    def _ts_rocr(self, x, window):
        """
        ROCR.
        """
        return self._apply_xn_d([x], tb.ROCR, window, window-1)
        
    def _ts_rocp(self, x, window):
        """
        ROCP.
        """
        return self._apply_xn_d([x], tb.ROCP, window, window-1)

    def _ts_rocr_100(self, x, window):
        """
        ROCR100.
        """
        return self._apply_xn_d([x], tb.ROCR100, window, window-1)

    def _obv(self, x1, x2):
        """
        OBV.
        """
        def func(df_adj):
            return tb.OBV(df_adj['x1'].values, df_adj['x2'].values)
        return self._apply_x1x2(x1, x2, func)

    def _ts_rsi(self, x, window):
        """
        Relative Strenght Index (RSI).
        """
        return self._apply_xn_d([x], tb.RSI, window, window-1)

    def _ts_dx(self, x1, x2, x3, window):
        """
        DX.
        """
        dx_re = self._apply_xn_d([x1, x2, x3], tb.DX, window, window-1)
        return dx_re
        # return self._apply_xn_d([x1, x2, x3], tb.DX, window, window-1)

    def _ts_minus_di(self, x1, x2, x3, window):
        """
        MINUS_DI.
        """
        return self._apply_xn_d([x1, x2, x3], tb.MINUS_DI, window, window-1)

    def _ts_plus_di(self, x1, x2, x3, window):
        """
        PLUS_DI.
        """
        return self._apply_xn_d([x1, x2, x3], tb.PLUS_DI, window, window-1)

    def _trange(self, x1, x2, x3):
            """
            TRANGE.
            """
            return self._apply_xn([x1, x2, x3], tb.TRANGE)

    def _avgprice(self, x1, x2, x3, x4):
        """
        AVGPRICE.
        """
        return self._apply_xn([x1, x2, x3, x4], tb.AVGPRICE)

    def _medprice(self, x1, x2):
        """
        MEDPRICE.
        """
        return self._apply_xn([x1, x2], tb.MEDPRICE)

    def _typprice(self, x1, x2, x3):
        """
        TYPPRICE.
        """
        return self._apply_xn([x1, x2, x3], tb.TYPPRICE)

    def _wclprice(self, x1, x2, x3):
        """
        WCLPRICE.
        """
        return self._apply_xn([x1, x2, x3], tb.WCLPRICE)

    def _ts_tbbeta(self, x1, x2, window):
        """贝塔系数"""
        return self._apply_xn_d([x1, x2], tb.BETA, window, window-1)

    def _ts_correl(self, x1, x2, window):
        """皮尔逊相关系数"""
        return self._apply_xn_d([x1, x2], tb.CORREL, window)

    def _ts_linearreg(self, x, window):
        """线性回归"""
        return self._apply_xn_d([x], tb.LINEARREG, window)

    def _ts_linerreg_angle(self, x, window):
        """线性回归的角度"""
        return self._apply_xn_d([x], tb.LINEARREG_ANGLE, window)

    def _ts_tsf(self, x, window):
        """时间序列预测"""
        return self._apply_xn_d([x], tb.TSF, window)

    def _ts_var(self, x, window):
        """方差"""
        return self._apply_xn_d([x], tb.VAR, window)

    def _atan(self, x):
        """反正切值"""
        return self._apply_xn([x], tb.ATAN)

    def _ceil(self, x):
        """向上取整数"""
        return self._apply_xn([x], tb.CEIL)

    def _cosh(self, x):
        """双曲正弦函数"""
        return self._apply_xn([x], tb.COSH)

    def _exp(self, x):
        """指数曲线"""
        return self._apply_xn([x], tb.EXP)

    def _floor(self, x):
        """向下取整数"""
        return self._apply_xn([x], tb.FLOOR)

    def _ln(self, x):
        """自然对数"""
        return self._apply_xn([x], tb.LN)

    def _log10(self, x):
        """对数函数log"""
        return self._apply_xn([x], tb.LOG10)

    def _sinh(self, x):
        """双曲正弦函数"""
        return self._apply_xn([x], tb.SINH)

    def _tan(self, x):
        """对数函数log"""
        return self._apply_xn([x], tb.TAN)

    def _ts_minus_dm(self, x1, x2, window):
        """
        MINUS_DM.
        """
        return self._apply_xn_d([x1, x2], tb.MINUS_DM, window)

    def _ts_plus_dm(self, x1, x2, window):
        """
        PLUS_DM.
        """
        return self._apply_xn_d([x1, x2], tb.PLUS_DM, window)


    # fintness
    def _rank_ic(self, x: pd.Series, thread_20, thread_80):
        '''rank ic index'''
        df = pd.DataFrame({'y_pred': x}).dropna()
        df['y_sig'] = np.where(df['y_pred']>thread_80, 1, 0)
        df['y_sig'] = np.where(df['y_pred']<thread_20, -1, df['y_sig'])
        df.dropna(inplace=True)
        df['y_sig'].replace(0, None, inplace=True)
        df['y_sig'].fillna(method='ffill', inplace=True)
        return df['y_sig']

    def _total_return(self, x: pd.Series):
        '''总收益率'''
        def get_sig(x: pd.Series):
            thread_20, thread_80 = x.quantile(.2), x.quantile(.8)
            x_v = x.iloc[-1]
            if x_v > thread_80:
                sig = 1
            elif x_v < thread_20:
                sig = -1
            else:
                sig = 0
            return sig
        x =x.rolling(70).apply(get_sig)
        x.replace(0, None, inplace=True)
        x.fillna(method='ffill', inplace=True)
        return x
            
    def _total_return_all_quantile(self, x: pd.Series, thread_20, thread_80):
        '''总收益率 总分位数求信号'''
        df = pd.DataFrame({'y_pred': x}).dropna()
        df['y_sig'] = np.where(df['y_pred']>thread_80, 1, 0)
        df['y_sig'] = np.where(df['y_pred']<thread_20, -1, df['y_sig'])
        df.dropna(inplace=True)
        df['y_sig'].replace(0, None, inplace=True)
        df['y_sig'].fillna(method='ffill', inplace=True)
        # print(df['y_sig'].iloc[-1])
        return df['y_sig']



class FactorIndex(AdjIndex):
    '''pandas_ta和talib'''
    def __init__(self, df):
        super().__init__(df)
        self.pdtb = ['aberration', 'above', 'accbands', 'ad', 'adosc',
                'alma', 'amat', 'ao', 'aobv', 'bbands', 'below', 
                'bias', 'cdl_pattern', 'cdl_z', 
                'cfo', 'cg', 'chop', 'cksp', 'cmf', 'coppock', 'cross', 'cross_value', 
                'cti', 'decay', 'decreasing', 'dm', 'donchian', 'dpo', 'ebsw', 'efi',
                'entropy', 'eom', 'er', 'eri', 'fisher', 'fwma', 'ha', 'hilo', 'hl2', 
                'hlc3', 'hma', 'hwma', 'ichimoku', 'increasing', 'inertia', 'jma', 
                'kama', 'kc', 'kst', 'kurtosis', 'kvo', 'linreg', 'log_return', 
                'mad', 'massi', 'mcgd', 'median', 'mfi', 'kdj', 
                'nvi', 'obv', 'ohlc4', 'pdist', 'percent_return', 'pgo',
                'psar', 'psl', 'pvi', 'pvo', 'pvol', 'pvr', 'pvt', 'pwma', 'qqe', 'qstick', 
                'quantile', 'rma', 'rsx', 'rvgi', 'rvi', 'sinwma', 
                'skew', 'slope', 'smi', 'squeeze', 'squeeze_pro', 'ssf', 'stc', 'stdev', 
                'stoch', 'stochrsi', 'supertrend', 'swma', 'td_seq', 'tema', 'thermo', 
                'tos_stdevall', 'true_range', 'tsi', 'tsignals', 'ttm_trend', 
                'ui', 'uo', 'variance', 'vhf', 'vidya', 'vortex', 'vp', 'vwap', 'vwma', 'wcp', 
                'willr', 'wma', 'xsignals', 'zlma', 'zscore']

        self.no_stationary_li = ['THERMO_', 'THERMOma_', 'VTXP_', 'DMP_', 'DMN_', 'VTXM_',
                                 'BBB_', 'BBP_']
    
    def madifrsi(self, n1=12, n2=46, n3=10):
        '''ma(rsi_fast-rsi_low)'''
        rsi_fast = self.rsi(n1)
        rsi_low = self.rsi(n2)
        ma_dif_rsi = tb.SMA(rsi_fast-rsi_low, n3)
        return ma_dif_rsi

    def pandas_ta(self, df, n, index_name):
        index_dic = {}
        try:
            n_i = [n] if not isinstance(n, list) else n
            df_i = getattr(df.ta, index_name)(*n_i)
            li = df_i.columns.to_list()
            index_dic.update({index_name: li})
            # print('df_i:', index_name, li)
            column = [f'{i}_{n}' for i in li]
            df[column] = df_i
            # 对非平稳序列进行修改
            # df = self.__adj_index(df, column)
        except:
            column = ''
        return df, column
    
    def __adj_index(self, df: pd.DataFrame, column: list):
        '''调整指标为平稳序列'''
        # if len(filter_str('VTXP', column, is_list=1)) or len(filter_str('DMP', column, is_list=1)):
        #     df[column[0]] = df[column[0]] - df[column[1]]
        #     del df[column[1]]
        # else:
        for col in column:
            ns_li = filter_str(col.split('_')[0], self.no_stationary_li, is_list=1)
            if len(ns_li): df[col] = df[col] / df['close']
        return df


class MyFactorIndex(AdjIndex, MyIndex):
    def __init__(self, df):
        AdjIndex.__init__(self, df)
        MyIndex.__init__(self, df)

        self.pdtb = ['aberration', 'above', 'accbands', 'ad', 'adosc',
                'alma', 'amat', 'ao', 'aobv', 'bbands', 'below', 
                'bias', 'cdl_pattern', 'cdl_z', 
                'cfo', 'cg', 'chop', 'cksp', 'cmf', 'coppock', 'cross', 'cross_value', 
                'cti', 'decay', 'decreasing', 'dm', 'donchian', 'dpo', 'ebsw', 'efi',
                'entropy', 'eom', 'er', 'eri', 'fisher', 'fwma', 'ha', 'hilo', 'hl2', 
                'hlc3', 'hma', 'hwma', 'ichimoku', 'increasing', 'inertia', 'jma', 
                'kama', 'kc', 'kst', 'kurtosis', 'kvo', 'linreg', 'log_return', 
                'mad', 'massi', 'mcgd', 'median', 'mfi', 'kdj', 
                'nvi', 'obv', 'ohlc4', 'pdist', 'percent_return', 'pgo',
                'psar', 'psl', 'pvi', 'pvo', 'pvol', 'pvr', 'pvt', 'pwma', 'qqe', 'qstick', 
                'quantile', 'rma', 'rsx', 'rvgi', 'rvi', 'sinwma', 
                'skew', 'slope', 'smi', 'squeeze', 'squeeze_pro', 'ssf', 'stc', 'stdev', 
                'stoch', 'stochrsi', 'supertrend', 'swma', 'td_seq', 'tema', 'thermo', 
                'tos_stdevall', 'true_range', 'tsi', 'tsignals', 'ttm_trend', 
                'ui', 'uo', 'variance', 'vhf', 'vidya', 'vortex', 'vp', 'vwap', 'vwma', 'wcp', 
                'willr', 'wma', 'xsignals', 'zlma', 'zscore']
    
    def madifrsi(self, n1=12, n2=46, n3=10):
        '''ma(rsi_fast-rsi_low)'''
        rsi_fast = self.rsi(n1)
        rsi_low = self.rsi(n2)
        ma_dif_rsi = tb.SMA(rsi_fast-rsi_low, n3)
        return ma_dif_rsi

    def pandas_ta(self, df, n, index_name):
        index_dic = {}
        try:
            n_i = [n] if not isinstance(n, list) else n
            df_i = getattr(df.ta, index_name)(*n_i)
            li = df_i.columns.to_list()
            index_dic.update({index_name: li})
            # print('df_i:', index_name, li)
            column = [f'{i}_{n}' for i in li]
            df[column] = df_i
        except:
            column = ''
        return df, column


class FactorIndexStatistics(FactorIndex):
    '''指标统计，统计每个因子属于哪个指标'''
    def __init__(self, df=None):
        if df is None:
            df = pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
        super().__init__(df)
    
    def save_pandas_index_name(self):
        '''保存pandas_tb指标'''
        index_dic = {}
        n = 10
        for index_name in self.pdtb:
            try:
                df_i = getattr(self.df.ta, index_name)(n)
                li = df_i.columns.to_list()
                index_dic.update({index_name: li})
                # print('df_i:', index_name, li)
                column = [f'{i}_{n}' for i in li]
                self.df[column] = df_i
            except:
                column = ''
        df_index = dic_to_dataframe(index_dic)
        pa = f'{pa_prefix}/datas/factor_info/'
        makedir(pa)
        df_index.to_csv(f'{pa}pandas_index_info.csv', index=False)
        print('save_pandas_index_name done.')
    
    def revise_index_name(self, df: pd.DataFrame):
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
        return df

    def index_category(self, load_pa):
        '''对指标分类
        df: columns, index_i, win_n, index_name, index_category
        '''
        df_pandas_index = pd.read_csv(f'{pa_prefix}/datas/factor_info/pandas_index_info.csv')
        df_index = pd.read_csv(f'{load_pa}.csv').iloc[1:]
        category_li, pandas_talib, win_li, index_li = [], [], [], []
        for i in range(df_index.shape[0]):
            index_i = df_index['columns'].iloc[i]
            win_li.append(eval(index_i.split('_')[-1]))
            index_i = get_sy(df_index['columns'].iloc[i])[:-1]
            index_li.append(index_i)
            ta_lib = 1
            for j in df_pandas_index.columns:
                if len(list(filter(lambda x: f'{index_i}_' in x, df_pandas_index[j].astype(str).to_list()))):
                    category_li.append(j)
                    pandas_talib.append('pandas_tb')
                    ta_lib = 0
                    break
            if ta_lib:
                category_li.append(index_i)
                pandas_talib.append('ta_lib')

        df_index['index_i'] = index_li
        df_index['win_n'] = win_li
        df_index['index_name'] = category_li
        df_index['index_category'] = pandas_talib

        # 修正
        # df_index = self.revise_index_name(df_index.copy())

        df_index.to_csv(f'{load_pa}_adj.csv', index=False)
        return df_index

    def caculate_select_index(self, index_pa, df=None):
        '''计算所选的技术指标'''
        if df is None:
            df = self.df.copy()
        column_orig = df.columns.to_list()[1:]
        if isinstance(index_pa, str):
            df_index = pd.read_csv(f'{index_pa}.csv')
        else:
            df_index = index_pa.copy()
        for i in range(len(df_index)):
            _, index_i, win_n, index_name, index_category = df_index.iloc[i]
            if index_category == 'pandas_tb':
                df_i = getattr(df.ta, index_name)(win_n)
                li = list(filter(lambda x: index_i in x, df_i.columns.to_list()))
                column = [f'{i}_{win_n}' for i in li]
                df[column] = df_i[li]
            elif index_category == 'ta_lib':
                func = getattr(self, index_name)
                params_n = func.__code__.co_argcount    # 判断指标需要输入的参数
                if params_n == 1:
                    res = func()
                else:
                    res = func(win_n)
                df[f'{index_i}_{win_n}'] = res if type(res) is not tuple else res[0]
            else:
                print('wrong inter')
        df.drop(columns=column_orig, inplace=True)
        return df


def run_loadfactors():
    '''下载期货因子'''
    startdate = '20130401'
    enddate = '20211102'
    interval = '1d'
    loadfactors = LoadFactors(startdate, enddate, interval)
    re = loadfactors.multi_progress_save_factors()
    re.to_csv('re.csv')
    print(re)

def FactorIndexStatistics_debug():
    df = pd.read_csv(f'{pa_prefix}/datas/data_60m/M/M1701.csv')
    load_pa = f'{pa_prefix}/filter_results/v/res5/[5, 0.5, 1, 1]_v_60m_1.2_20_1_return_rate_60m_col_adj'
    f = FactorIndexStatistics(df)
    print(f.caculate_select_index(load_pa).tail(5))

def gplearnindex_debug():
    df = pd.read_csv(f'{pa_prefix}/datas/data_60m/M/M1701.csv')
    index_name = 'ts_decay_linear(ts_decay_linear(close, 4), 4)'
    gpl = GPLearnIndex()
    gpl.set_datas(df)
    df['index1'] = gpl.get_index(index_name)
    df.to_csv('df.csv')
    print(df.head(30))
    print('---------')
    print(df.tail(30))

if __name__ == "__main__":
    # run_loadfactors()
    
    gplearnindex_debug()
    
    # transform_factors()
    # startdate = '20130401'
    # enddate = '20211102'
    # interval = '1d'
    # loadfactors = LoadFactors(startdate, enddate, interval)
    # df = loadfactors.get_rqdatas()
    # df = loadfactors.get_datayes()
    # df.to_csv('df0.csv')
    # df= loadfactors.factor_example()
    # print(df)

    # res = LoadFactors.multi_progress_save_factors()
    # df = lf.datas_process(industry_index=1)
    # print(df)
    

    # startdate = 20130401
    # enddate = 20211021
    # interval = '1d'
    # loadfactors = LoadFactors(startdate, enddate, interval)
    # # price = loadfactors.get_factors('ZC88')
    # # # price = price.iloc[:, 1:].apply(m_format)
    # # # price = loadfactors.get_price('ZC')
    # price = loadfactors.get_wind('J')
    # price = w.wset("productsfund","startdate={};enddate={};product={}; \
    #         field=date,holdnumber,holdnumberchange,holdnumber_margin,holdnumberchange_margin".format('20201011', '20201111', 'ZC'), usedf=True)[1]
    
    # # factor_example = loadfactors.factor_example()
    # print(price)

    # factor_example.to_csv('df.csv')
    # col = factor_example.columns.tolist()
    # s = pd.DataFrame()
    # s['e'] = col
    # print(s)

    # s.to_csv('s.csv')


    
# %%
