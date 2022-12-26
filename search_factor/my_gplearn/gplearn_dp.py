from array import array
from cgi import print_directory
import sys, os
from turtle import right
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
from datetime import datetime
from datas_process.m_futures_factors import MainconInfo, SymbolsInfo

# import pandas as pd
__Author__ = 'ZCXY'
# s = pd.DataFrame()
# # s['r'] = [1,3,5,2,6,7]
# # print(pd.DataFrame(s['r'])['r'])
# s1 = pd.DataFrame()
# s1['r'] = [9,11,5,3,6,7]
# d = pd.merge(s, s1, how='outer', left_on='', right_on='r')
# d

def generate_all_maincon_table(symbol_li=None, return_n=1, startdate=None, enddate=None, is_normal=0):
    '''获取合约k线合在一张表里
    return:
                                      open    high   low   close   volume
    datetime                contract       
    2016-01-01 21:00:00     RB1606    3000    3000   3000  3000    3000
                            RB2110    3000    3000   3000  3000    3000
                            FG1605    3000    3000   3000  3000    3000
                            .
                            .
                            .
    2016-01-01 22:00:00     RB1606    3000    3000   3000  3000    3000
                            RB2110    3000    3000   3000  3000    3000
                            FG1605    3000    3000   3000  3000    3000
                            .
                            .
                            .
    '''
    save_pa = f'{pa_prefix}/search_factor/my_gplearn/raw_data/'
    suffix = f'_normal_{is_normal}' if is_normal else ''  # 1 close/pre_close 2 close/close0
    mi, si = MainconInfo(), SymbolsInfo()
    symbol_dic = {}
    # symbol_li = ['RB', 'RU', 'M', 'AP', 'AL']
    if symbol_li is None:
        symbol_li = si.symbol_li
    if startdate is None:
        startdate = datetime(2016, 1, 1)
    if enddate is None:
        enddate = datetime(2019, 5, 1)
    
    df_all_contract_li = []
    for sy in symbol_li:    # 将所有时间段主力合约全部提取出来
        # print(sy)
        df_li = mi.get_main_contact_k_line(sy, startdate, enddate, delay=20, load_pa=None, is_concat=0, contract_name=1)
        df_all_contract_li = df_all_contract_li + df_li
        symbol_dic[sy] = [len(df_all_contract_li)]

    contract_n = len(df_all_contract_li)
    pd.DataFrame(symbol_dic).to_csv(f'{save_pa}df_contract_count.csv', index=False)
    
    df_dt= pd.DataFrame()
    df_dt['datetime'] = []
    for df_i in df_all_contract_li:     # 获取交易时间段的并集
        df_dt = pd.merge(df_dt, pd.DataFrame(df_i['datetime']), how='outer', left_on='datetime', right_on='datetime')
    
    df_dt.sort_values(by='datetime', ascending=True, inplace=True)
    df_dt.to_csv(f'{save_pa}df_dt.csv')
    dt_n = len(df_dt)

    df_y = pd.DataFrame()
    df_y['datetime'] = df_dt['datetime'].to_list()
    df_concat_li = []
    for df_i in df_all_contract_li:     # 将所有主力合约拼在一张表里
        del df_i['turnover']
        contract = df_i['contract'].iloc[0]
        df_i['y'] = df_i['close'].pct_change(return_n).shift(-return_n)
        if is_normal == 1:
            df_i['pre_close'] = df_i['close'].shift(1)
            df_i['open'] = df_i['open'] / df_i['pre_close']
            df_i['high'] = df_i['high'] / df_i['pre_close']
            df_i['low'] = df_i['low'] / df_i['pre_close']
            df_i['close'] = df_i['close'] / df_i['pre_close']
            df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)
            del df_i['pre_close']

        elif is_normal == 2:
            df_i['open'] = df_i['open'] / df_i['open'].shift(1)
            df_i['high'] = df_i['high'] / df_i['high'].shift(1)
            df_i['low'] = df_i['low'] / df_i['low'].shift(1)
            df_i['close'] = df_i['close'] / df_i['close'].shift(1)
            df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)

        elif is_normal == 3:
            df_i['open'] = df_i['open'] / df_i['open'].shift(1)
            df_i['high'], df_i['low'] = df_i['high'] / df_i['low'], df_i['low'] / df_i['high']
            df_i['close'] = df_i['close'] / df_i['close'].shift(1)
            df_i['volume'] = df_i['volume'] / df_i['volume'].shift(1)

        df_adj = pd.merge(df_i, df_dt, how='outer', left_on='datetime', right_on='datetime')
        df_adj.sort_values('datetime', ascending=True, inplace=True)
        df_adj['contract'] = contract
        df_y_i = df_adj[['datetime', 'y']]
        df_y_i.columns = ['datetime', contract]
        df_y = pd.merge(df_y, df_y_i, left_on='datetime', right_on='datetime', how='outer')
        df_adj.pop('y')
        df_concat_li.append(df_adj.copy())
    
    df_concat = pd.concat(df_concat_li)
    df_concat.set_index('datetime', inplace=True)
    df_concat.set_index('contract', inplace=True, append=True)
    df_concat.sort_index(level=0, inplace=True)

    df_y.set_index('datetime', inplace=True)
    df_y = df_y.T.sort_index().T
    df_y.to_csv(f'{save_pa}y{suffix}.csv')
    df_concat.to_csv(f'{save_pa}x{suffix}.csv')
    arr_y = df_y.T.values
    arr_x = df_concat.values.reshape((dt_n, contract_n, -1))

    print(arr_y.shape, arr_x.shape)

    np.save(f'{save_pa}x{suffix}.npy', arr_x)
    np.save(f'{save_pa}y{suffix}.npy', arr_y)
    

if __name__ == '__main__':
    generate_all_maincon_table(is_normal=3)


    




    

    


