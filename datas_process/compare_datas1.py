import pandas as pd
__Author__ = 'ZCXY'
import sys, os
from m_base import *
sys_name = 'windows'
pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import makedir


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


if __name__ == '__main__':
    interval = 1
    for interval in [1, 60]:
        pa1 = f'{pa_prefix}/datas1/data_{interval}m/'
        pa2 = f'{pa_prefix}/datas_rq/data_{interval}m/'
        save_pa = f'{pa_prefix}/datas/datas_diff1/data_{interval}m'
        compare_datas(pa1, pa2, save_pa)
