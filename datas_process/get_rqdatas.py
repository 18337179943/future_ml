import imp
import rqdatac
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.'
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import get_database
from datas_process.m_futures_factors import SymbolsInfo
from m_base import get_sy, makedir

rqdatac.init('18337179943','123456')

class BaseGetRqdatas:
    '''获取米筐数据的基类'''
    def __init__(self, startdate='20211101', enddate='20220816'):
        self.startdate = startdate
        self.enddate = enddate
        sy = SymbolsInfo()
        self.symbols = sy.symbols
        self.save_pa = './datas_rq/'
        self.frequency = '1d'
        self.data_type = 0
        if not os.path.exists(self.save_pa):
            os.makedirs(self.save_pa)

    def get_maincon(self):
        '''获取每日主力合约'''

        pa = self.save_pa+'maincon.csv'
        # try:
        #     res = pd.read_csv(pa)
        #     res['date'] = pd.to_datetime(res['date'])
        # except:
        res = pd.DataFrame()
        for symbol in self.symbols:
            df = pd.DataFrame(rqdatac.futures.get_dominant(symbol, self.startdate, self.enddate))
            df.columns = ['contract']
            df['contract1'] = df['contract'].shift(-1)
            df['is_change'] = np.where(df['contract1']!=df['contract'], 1, 0)
            df['symbol'] = symbol
            res = pd.concat([res, df])
        res.to_csv(pa)
        res.reset_index(inplace=True)
        return res
    

class GetRqdatas(BaseGetRqdatas):
    '''获取主连数据'''
    def __init__(self, startdate='20211101', enddate='20220816'):
        super().__init__(startdate, enddate)

    def get_rqdatas(self, symbol, save=1):
        '''获取米筐k线数据'''
        print(symbol, 'begin.')
        if self.data_type == 0:
            symbol = str.upper(symbol) + '88'
            df = rqdatac.get_price(symbol, start_date=self.startdate, end_date=self.enddate, frequency=self.frequency).reset_index('order_book_id')
            if self.frequency == '1d':
                df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'open_interest']]
            else:
                df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'open_interest', 'trading_date']]
        else:
            df = rqdatac.futures.get_dominant_price(symbol, start_date=self.startdate, end_date=self.enddate, frequency=self.frequency, 
                                                    adjust_type='pre', adjust_method='prev_close_ratio')
            # del df['underlying_symbol']
        df['pct_change'] = (df['close']-df['open'])/df['open']
        df['high_low_pct'] = (df['high']-df['low'])/df['open']
        df['symbol'] = symbol
        if save:
            df.to_csv(self.save_pa+'datas_'+self.frequency+'/'+symbol+'.csv', encoding="utf_8_sig")
        print(symbol, 'done.')
        return df
    
    def multi_progress_load_datas(self, frequency):
        '''多进程下载k线数据'''
        self.frequency = frequency
        if not os.path.exists(self.save_pa+'datas_'+self.frequency+'/'):
            os.makedirs(self.save_pa+'datas_'+self.frequency+'/')
        with ProcessPoolExecutor(max_workers=3) as executor:  # max_workers=10
            res = executor.map(self.get_rqdatas, self.symbols)
        if self.frequency == '1d':
            res = pd.concat([i for i in res])
            res.to_csv('./datas_rq/datas_1d.csv')
        print('done.')
        return 0

    def save_datas_to_sql(self, pa=None):
        '''将数据保存到数据库里'''
        if pa == None:
            pa_list = os.listdir(self.save_pa+'datas_'+self.frequency+'/')
        # else:
        #     pa_list = os.listdir(pa)
        for pa in pa_list: 
            # try:
            bars = []
            data_df = pd.read_csv(self.save_pa + pa)
            data_df.dropna(inplace=True)
            data_df['datetime'] = pd.to_datetime(data_df['datetime'])
            data_df['datetime'] = data_df['datetime'].apply(lambda x: x-timedelta(hours=8)-timedelta(minutes=1))
            symbol = pa.split('.')[0]
            data_list = data_df.to_dict('records')
            for item in data_list:
                dt = datetime.fromtimestamp(item['datetime'].timestamp())
                bar = BarData(
                    symbol=symbol,
                    exchange=Exchange.LOCAL,
                    datetime=dt,  # datetime.fromtimestamp(item['datetime'].timestamp()),
                    interval=Interval.MINUTE,
                    open_price=float(item['open']),
                    high_price=float(item['high']),
                    low_price=float(item['low']),
                    close_price=float(item['close']),
                    volume=float(item['volume']),
                    gateway_name="DB",
                )
                bars.append(bar)
            database_manager = get_database()
            database_manager.save_bar_data(bars)
            print(pa, 'done.')

    def run(self):
        '''执行下载数据'''
        self.data_type = 0
        self.multi_progress_load_datas('1m')
        self.save_datas_to_sql()

        # self.get_maincon()

        self.data_type = 1
        self.multi_progress_load_datas('60m')


class GetAllRqdatas(BaseGetRqdatas):
    '''获取所有主力合约历史数据'''
    def __init__(self, startdate='20211101', enddate='20220816'):
        super().__init__(startdate, enddate)
        self.df_maincon = self.get_maincon()
        self.pa = './datas/data_1min/'

        if not os.path.exists(self.save_pa):
            os.makedirs(self.save_pa)

    def get_rqdatas(self, contract, startdate, enddate):
        '''获取米筐数据'''
        df = rqdatac.get_price(contract, start_date=startdate, end_date=enddate, frequency=self.frequency).reset_index('order_book_id')
        if self.frequency == '1d':
            df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'open_interest']]
        else:
            df = df[['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'open_interest', 'trading_date']]
        return df

    def save_rqdatas(self, df, save_pa, file_name):
        '''保存历史数据'''
        makedir(save_pa)
        df.to_csv(save_pa+file_name+'.csv')
    
    def load_datas(self, frequency='1m'):
        contracts = self.df_maincon['contract'].unique()
        self.frequency = frequency
        pa = self.pa
        # pa = self.pa + frequency + '/'
        # makedir(pa)
        for contract in contracts:
            print(contract)
            df = self.df_maincon[self.df_maincon['contract']==contract]
            startdate = df['date'].iloc[0].date() - timedelta(days=30)
            enddate = df['date'].iloc[-1].date() + timedelta(days=20)
            df = self.get_rqdatas(contract, startdate, enddate)
            symbol = get_sy(contract)
            self.save_rqdatas(df, pa+symbol+'/', contract)
            # print(contract, symbol, startdate, enddate)


    

if __name__ == '__main__':
    gr = GetAllRqdatas()
    gr.load_datas('1m')

    # gr = GetRqdatas()
    # gr.get_rqdatas('RB').tail(20)
    # gr.run()
    # gr.get_maincon()
    # gr.get_rqdatas('A')
    # gr.save_datas_to_sql('./datas/datas_60m_88')
    # re = gr.get_maincon()
    # gr.multi_progress_load_datas('60m')
    # gr.get_rqdatas('RB')
    



