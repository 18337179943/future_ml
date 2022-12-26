import pandas as pd

import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.'
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
from m_base import makedir
from datetime import timedelta, time


def compare_datas(pa1, pa2, save_pa):
    symbol_li = os.listdir(pa1)
    makedir(save_pa)
    for symbol in symbol_li:
        try:
            contract_li = os.listdir(f'{pa1}{symbol}')
            for contract in contract_li:
                df1 = pd.read_csv(f'{pa1}{symbol}/{contract}')
                df2 = pd.read_csv(f'{pa2}{symbol}/{contract}')
                df_merge = pd.merge(df1, df2, left_on='datetime', right_on='datetime', how='outer')
                df_diff = pd.DataFrame()
                df_diff['datetime'] = df_merge['datetime']
                df_diff['open'] = df_merge['open_x'] - df_merge['open_y']
                df_diff['high'] = df_merge['high_x'] - df_merge['high_y']
                df_diff['low'] = df_merge['low_x'] - df_merge['low_y']
                df_diff['close'] = df_merge['close_x'] - df_merge['close_y']
                df_diff['volume'] = df_merge['volume_x'] - df_merge['volume_y']
                save_pa_i = makedir(f'{save_pa}/{symbol}/')
                df_diff.to_csv(f'{save_pa_i}{contract}', index=False)
                print(contract, 'done.')
        except:
            print('have problem', symbol)
    return

def compare_single_datas(pa1, pa2, save_pa):
    df1 = pd.read_csv(pa1)
    df2 = pd.read_csv(pa2)
    df2.dropna(inplace=True)
    df2.rename(columns={'LastPrice_open': 'open', 'LastPrice_high': 'high', 'LastPrice_low': 'low', 'LastPrice_close': 'close', 'Volume': 'volume', 'Turnover': 'turnover',
        'Date': 'datetime'}, inplace=True)
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    df2['datetime'] = pd.to_datetime(df2['datetime'])
    def change_datetime_sc(x):
        '''转换双璨数据'''
        if x.time() == time(12, 30) or x.time() == time(10, 30):
            x = datetime(x.year, x.month, x.day, 11, 0)
        if x.minute < 5:
            x = datetime(x.year, x.month, x.day, x.hour, 0)
        return x
    df2['datetime'] = df2['datetime'].apply(lambda x: x-timedelta(hours=1))
    df2['datetime'] = df2['datetime'].apply(change_datetime_sc)
    df_merge = pd.merge(df1, df2, left_on='datetime', right_on='datetime', how='outer')
    # df_diff = pd.DataFrame()
    df_merge['datetime_d'] = df_merge['datetime']
    df_merge['open_d'] = df_merge['open_x'] - df_merge['open_y']
    df_merge['high_d'] = df_merge['high_x'] - df_merge['high_y']
    df_merge['low_d'] = df_merge['low_x'] - df_merge['low_y']
    df_merge['close_d'] = df_merge['close_x'] - df_merge['close_y']
    df_merge['volume_d'] = df_merge['volume_x'] - df_merge['volume_y']
    df_merge.to_csv(save_pa)
    return

def run_compare_datas():
    interval = 1
    for interval in [1, 60]:
        # pa1 = f'{pa_prefix}/datas/data_{interval}m/'
        pa1 = f'{pa_prefix}/datas/data_{interval}m/'
        pa2 = f'{pa_prefix}/datas_adj/data_{interval}m/'
        save_pa = f'{pa_prefix}/datas/datas_diff/data_{interval}m'
        compare_datas(pa1, pa2, save_pa)

def run_single_compare_datas():
    contract = 'RB2301'
    pa1 = f'{pa_prefix}/datas/compare_datas/{contract}.csv'
    pa2 = f'{pa_prefix}/datas/compare_datas/{contract}sc.csv'
    save_pa = f'{pa_prefix}/datas/compare_datas/{contract}_diff.csv'
    compare_single_datas(pa1, pa2, save_pa)

if __name__ == '__main__':
    run_single_compare_datas()
