from datetime import datetime
import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
import os
import warnings
warnings.filterwarnings("ignore")

from functools import reduce

from data_process import *

'''
品种持仓: 
1、缺少SC

会员持仓（成交量、持仓买量、持仓卖量）:
1、缺少SC
2、三个表交集差异集中在2014-2016的SF

缺失因子用均值填充
'''


class RawData:
    def __init__(self) -> None:
        self.contract_list = self.contract()
        # self.start = '20210101'
        self.start = '20160101'
        self.end = datetime.now().strftime('%Y%m%d')
        if not os.path.exists('data'):
            os.makedirs('data')
    
    def get_future_sectors(self, obj=None):
        '''
        obj:    =None: return all dict
                ='hc': return 'black'
        '''
        future_sectors = {'finance': ['IF', 'IC', 'IH', 'T', 'TF'],
                        'black': ['hc', 'i', 'j', 'jm', 'rb', 'SF', 'ZC'],
                        'chemical': ['eg', 'FG', 'l', 'MA', 'pp', 'TA', 'v'],
                        'energy': ['bu', 'pg', 'sc'],
                        'agricultural': ['a', 'AP', 'c', 'cs', 'jd', 'm', 'OI', 'p', 'RM', 'y'],
                        'soft': ['CF', 'ru', 'SR'],
                        'nonferrous': ['ag', 'al', 'au', 'cu', 'ni', 'pb', 'sn', 'zn']}
        if obj != None:
            future_sectors = {k:[i.upper() for i in v] for k,v in future_sectors.items()} # value改为大写
            return [k for k,v in future_sectors.items() if obj in v][0] # 根据品种返回板块名称
        return future_sectors

    def contract(self):
        contract = [v for k,v in self.get_future_sectors().items()]
        contract = reduce(lambda left, right: left+right, contract)
        contract = [i.upper() for i in contract]
        return contract
    
    def query_raw_volprice_data_myrule(self):
        '''初始量价因子(所有合约，非通联选取主力)'''
        sql = '''
            select a.TRADE_DATE date,
            a.CONTRACT_OBJECT object,
            a.TICKER_SYMBOL symbol,
            a.OPEN_PRICE open,
            a.CLOSE_PRICE close,
            a.HIGHEST_PRICE high,
            a.LOWEST_PRICE low,
            a.PRE_CLOSE_PRICE pre_close, 
            a.SETTL_PRICE settle,
            a.TURNOVER_VOL to_volume,
            a.TURNOVER_VALUE to_value,
            a.OPEN_INT interest,
            b.LIMIT_UP_PRICE,
            b.LIMIT_DOWN_PRICE,
            c.LAST_DELI_DATE last_deli_date,
            DATEDIFF(c.LAST_DELI_DATE, a.TRADE_DATE) datediff

            from mkt_futd a
            join mkt_fut_limit b 
            on a.SECURITY_ID=b.SECURITY_ID and a.TRADE_DATE=b.TRADE_DATE
            join futu c 
            on a.TICKER_SYMBOL = c.TICKER_SYMBOL

            where a.TRADE_DATE between {} and {}
            and DATEDIFF(c.LAST_DELI_DATE, a.TRADE_DATE) between 0 and 2000 
            /* 若不加此限制，会搜出10年前或10年后的合约 */

            order by a.TRADE_DATE,a.CONTRACT_OBJECT,a.TICKER_SYMBOL'''.format(self.start, self.end)
        df = query_sql_df(sql)
        df = df[df['object'].isin(self.contract_list)]
        df[['close', 'settle']] = df[['close', 'settle']].fillna(method='bfill', axis=1)
        df[['open', 'settle']] = df[['open', 'settle']].fillna(method='bfill', axis=1)
        df[['high', 'settle']] = df[['high', 'settle']].fillna(method='bfill', axis=1)
        df[['low', 'settle']] = df[['low', 'settle']].fillna(method='bfill', axis=1)
        df[['pre_close', 'settle']] = df[['pre_close', 'settle']].fillna(method='bfill', axis=1)
        df['to_value'].fillna(method='ffill', axis=0, inplace=True)
        return df


    # def query_open_int(self):
    #     '''
    #     品种多空持仓类数据
    #         ***
    #         通联数据大致从2016年开始
    #         缺少原油
    #         ***
    #     '''
    #     sql = '''
    #         select TRADE_DATE date, 
    #         CONTRACT_OBJECT object, 
    #         LONG_OPEN_INT long_oi, 
    #         SHORT_OPEN_INT short_oi, 
    #         RATIO oi_ratio 
    #         from mkt_fut_oi_ratio 
    #         where TRADE_DATE between {} and {} 
    #         order by TRADE_DATE
    #         '''.format(self.start, self.end)
    #     if not os.path.exists('data/openint/openint.csv'):
    #         df = query_sql_df(sql)
    #         df = df[df['object'].isin(self.contract_list)]
    #         df.to_csv('data/openint/openint.csv')
    #         return df
    #     else:
    #         df = pd.read_csv('data/openint/openint.csv', index_col=0)
    #         return df

    # def query_warehouse(self):
    #     '''仓单类数据'''
    #     sql = '''
    #         select TRADE_DATE date, 
    #         CONTRACT_OBJECT object, 
    #         sum(WR_VOL) wr_vol, 
    #         sum(CHG) chg_wr 
    #         from mkt_fut_wrd
    #         where TRADE_DATE between {} and {} 
    #         group by TRADE_DATE, CONTRACT_OBJECT 
    #         order by TRADE_DATE
    #         '''.format(self.start, self.end)
    #     if not os.path.exists('data/memrankoi/warehouse.csv'):
    #         df = query_sql_df(sql)
    #         df = df[df['object'].isin(self.contract_list)]
    #         df.to_csv('data/warehouse/warehouse.csv')
    #         return df
    #     else:
    #         df = pd.read_csv('data/warehouse/warehouse.csv', index_col=0)
    #         return df
    
    # def query_mem_rank_vol(self):
    #     '''会员成交量排名数据'''
    #     sql = '''
    #     select a.TRADE_DATE date,
    #     b.CONTRACT_OBJECT object,
    #     a.RANK mem_rank, 
    #     /*a.TICKER_SYMBOL symbol,*/
    #     /*a.PARTY_SHORT_NAME,*/
    #     a.TURNOVER_VOL mem_vol,
    #     a.CHG mem_vol_chg
    #     from mkt_futmtvr a
    #     join mkt_futd b on a.SECURITY_ID=b.SECURITY_ID and a.TRADE_DATE=b.TRADE_DATE
    #     where a.TRADE_DATE between {} and {} 
    #     and b.MAINCON=1 and a.RANK<=20
    #     '''.format(self.start, self.end)
    #     if not os.path.exists('data/memrankoi/memrk_vol.csv'):
    #         df = query_sql_df(sql)
    #         df = df[df['object'].isin(self.contract_list)]
    #         df.to_csv('data/memrankoi/memrk_vol.csv')
    #         return df
    #     else:
    #         df = pd.read_csv('data/memrankoi/memrk_vol.csv', index_col=0)
    #         return df
    
    # def query_mem_rank_longoi(self):
    #     '''会员持仓买量排名数据'''
    #     sql = '''
    #     select a.TRADE_DATE date,
    #     b.CONTRACT_OBJECT object,
    #     a.RANK mem_rank, 
    #     /*a.TICKER_SYMBOL symbol,*/
    #     /*a.PARTY_SHORT_NAME,*/
    #     a.LONG_OPEN_INT mem_long_oi,
    #     a.CHG mem_long_oi_chg
    #     from mkt_futmoibr a
    #     join mkt_futd b on a.TICKER_SYMBOL=b.TICKER_SYMBOL and a.TRADE_DATE=b.TRADE_DATE
    #     where a.TRADE_DATE between {} and {} 
    #     and b.MAINCON=1 and a.RANK<=20
    #     '''.format(self.start, self.end)
    #     if not os.path.exists('data/memrankoi/memrk_longoi.csv'):
    #         df = query_sql_df(sql)
    #         df = df[df['object'].isin(self.contract_list)]
    #         df.to_csv('data/memrankoi/memrk_longoi.csv')
    #         return df
    #     else:
    #         df = pd.read_csv('data/memrankoi/memrk_longoi.csv', index_col=0)
    #         return df

    # def query_mem_rank_shortoi(self):
    #     '''会员持仓卖量排名数据'''
    #     sql = '''
    #     select a.TRADE_DATE date,
    #     b.CONTRACT_OBJECT object,
    #     a.RANK mem_rank,
    #     /*a.TICKER_SYMBOL symbol,*/
    #     /*a.PARTY_SHORT_NAME,*/
    #     a.SHORT_OPEN_INT mem_short_oi,
    #     a.CHG mem_short_oi_chg
    #     from mkt_futmoiar a
    #     join mkt_futd b on a.TICKER_SYMBOL=b.TICKER_SYMBOL and a.TRADE_DATE=b.TRADE_DATE
    #     where a.TRADE_DATE between {} and {} 
    #     and b.MAINCON=1 and a.RANK<=20
    #     '''.format(self.start, self.end)
    #     if not os.path.exists('data/memrankoi/memrk_shortoi.csv'):
    #         df = query_sql_df(sql)
    #         df = df[df['object'].isin(self.contract_list)]
    #         df.to_csv('data/memrankoi/memrk_shortoi.csv')
    #         return df
    #     else:
    #         df = pd.read_csv('data/memrankoi/memrk_shortoi.csv', index_col=0)
    #         return df
