import pandas as pd
__Author__ = 'ZCXY'
import numpy as np



def run_exchange_time():
    symbol_dict = {
        'AG' : {'2013-07-06': {'hour': '2', 'min': '30'}},
        'AL' : {'2013-12-21': {'hour': '1', 'min': '00'}},
        'AP' : {},
        'BU' : {'2014-12-27': {'hour': '1', 'min': '00'}, '2016-05-03': {'hour': '23', 'min': '00'}},
        'C'  : {'2019-03-29':{'hour': '23', 'min':'00'}},
        'CF' : {'2019-12-11': {'hour': '23', 'min': '00'}, '2014-12-12': {'hour': '23', 'min': '30'}},
        'CU' : {'2013-12-21': {'hour': '1', 'min': '00'}},
        'FG' : {'2019-12-12': {'hour': '23', 'min': '00'}, '2015-08-31': {'hour': '23', 'min': '30'}},
        'HC' : {'2014-12-26': {'hour': '1', 'min': '00'}, '2016-05-03': {'hour': '23', 'min': '00'}},
        'I'  : {'2015-05-08':{'hour':'23','min':'30'},'2019-03-29':{'hour': '23', 'min':'00'}},     # 2014-04-27到2015-05-08到1点
        'J'  : {'2015-05-08':{'hour':'23','min':'30'},'2019-03-29':{'hour': '23', 'min':'00'}},     # 2014-07-07到2015-05-08到2点半
        'JD' : {},
        'JM' : {'2015-05-08':{'hour':'23','min':'30'},'2019-03-29':{'hour': '23', 'min':'00'}},     # 2014-12-26到2015-05-08到2点半
        'L'  : {'2019-03-29':{'hour': '23', 'min':'00'}},
        'M'  : {'2015-05-08':{'hour':'23','min':'30'},'2019-03-29':{'hour': '23', 'min':'00'}},        # 2014-12-27到2015-05-08到2点半
        'MA' : {'2019-12-12': {'hour': '23', 'min': '00'}, '2014-12-12': {'hour': '23', 'min': '30'}},  
        'NI' : {'2015-03-25': {'hour': '1', 'min': '00'}},
        'OI' : {'2019-12-11': {'hour': '23', 'min': '00'}, '2015-08-31': {'hour': '23', 'min': '30'}},
        'P'  : {'2015-05-08':{'hour':'23','min':'30'},'2019-03-29':{'hour': '23', 'min':'00'}},     # 2014-07-08到2015-05-08到2点半
        'PB' : {'2013-12-20': {'hour': '1', 'min': '00'}},
        'PP' : {'2019-03-29':{'hour': '23', 'min':'00'}},
        'RB' : {'2016-05-03':{'hour': '23', 'min':'00'}},
        'RM' : {'2019-12-12': {'hour': '23', 'min': '00'}, '2014-12-12': {'hour': '23', 'min': '30'}},
        'RU' : {'2015-01-05':{'hour': '23', 'min':'00'}},
        'SF' : {},
        'SM' : {},
        'SN' : {'2015-12-30': {'hour': '1', 'min': '00'}},
        'SR' : {'2019-12-12': {'hour': '23', 'min': '00'}, '2014-12-15': {'hour': '23', 'min': '30'}},
        'V'  : {'2019-03-29':{'hour': '23', 'min':'00'}},
        'Y'  : {'2015-05-08':{'hour':'23','min':'30'},'2019-03-29':{'hour': '23', 'min':'00'}},         # 2014-12-26到2015-05-08到2点半
        'ZC' : {'2015-08-31': {'hour': '23', 'min': '30'}, '2019-12-11': {'hour': '23', 'min': '00'}},
        'ZN' : {'2013-12-20':{'hour': '1', 'min':'00'}},
    }
    df = pd.read_csv('exchange_time.csv')

    # for symbol in symbol_dict:
    #     df.iloc[df[df['code']==symbol].index.to_list()[0]]['everning_end'] = symbol_dict[symbol]

    # df.to_csv('exchange_time1.csv', index=False)
    # print('run_exchange_time:', 'done')
    return df, symbol_dict


if __name__ == '__main__':
    night_symbols = ['AG', 'AU', 'CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'SC', 'BC', 'BU']
    symbols = ['AP', 'C', 'CF', 'FG', 'HC', 'I', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SF', 'SM', 'SR', 'V', 'Y', 'ZC']
    df, symbol_dict = run_exchange_time()
    li = list(symbol_dict.keys())
    li = list(filter(lambda x: x not in night_symbols, li))
    print(li)
    print(len(li))
