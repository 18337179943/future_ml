from pkgutil import ImpImporter
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
import os
import re
from datetime import timedelta
import joblib
import json
from collections import defaultdict
import multiprocessing
from group_by_type import run_group_by_type
from qh_marketdata import run_qh_marketdata

#from utils import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def get_data(save_path: str, contract_name: str, one_contract_list, margin, exchange_time, get_last_day):

    margin_list = [margin]*len(one_contract_list)
    exchange_time_list = [exchange_time]*len(one_contract_list)
    get_last_day_list = [get_last_day]*len(one_contract_list)

    pd_res = []
    
    for item in one_contract_list:
        print(item)
        pd_res.append(get_one_day(item, margin, exchange_time, get_last_day))
    #pd_res = list(map(get_one_day, one_contract_list, margin_list, exchange_time_list, get_last_day_list))
    res = pd.concat(pd_res)
    res.reset_index(drop=True, inplace=True)
    res.fillna(method='ffill', inplace=True)
    res.to_csv(save_path + '/' + contract_name +'.csv', index=False)
    #print(res)

    return 'success'


def get_one_day(file_name, margin, exchange_time, last_exchange_day):
    header = ['TradingDay', 'InstrumentID', 'UpdateTime', 'UpdateMillisec', 'LastPrice', 'Volume', 'BidPrice1',
              'BidVolume1', 'AskPrice1', 'AskVolume1', 'AveragePrice', 'Turnover', 'OpenInterest', 'UpperLimitPrice',
              'LowerLimitPrice']
    temp = pd.read_csv(file_name, header=None, names=header)
    day = pd.to_datetime(re.findall(r'_(\S+)\.', file_name))
    
    temp[['TradingDay']] = temp[['TradingDay']].astype(int)
    temp['Date'] = (temp['TradingDay'].map(str) + temp['UpdateTime'] + temp['UpdateMillisec'].map(str)).apply(lambda x: pd.to_datetime(x, format='%Y%m%d%H:%M:%S%f'))
    temp['Date'] = temp['Date'] + pd.Timedelta(milliseconds=1)
    if temp['AveragePrice'].isna().all():
        temp['AveragePrice'] = temp['Turnover']/temp['Volume']
    if len(temp) > 0 :
      everning_day = temp.loc[temp.index[0], 'Date']
    else:
      res = pd.DataFrame([], columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                        'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                        'LowerLimitPrice', 'Date', 'Avg_price'])
      return res  

    start1 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                             hour=9, minute=0, second=0)

    start2 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                             hour=10, minute=30, second=0)

    start3 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                             hour=13, minute=30, second=0)

    start_e = pd.Timestamp(year=everning_day.year, month=everning_day.month, day=everning_day.day,
                            hour=21, minute=0, second=0)

    end1 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                           hour=10, minute=15, second=0)

    end2 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                           hour=11, minute=30, second=0)

    end3 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                           hour=15, minute=0, second=0)
    # print(temp, start1,start2,start3,start3,end1,end2,end3)
    # 保留日盘第一个有效数据
    avail = temp[((temp['Date'] >= start1) & (temp['Date'] <= end3)) | (temp['Date'] >= start_e)]['LastPrice']
    #print(avail,111111)
    if len(avail) > 0:
      # 第一个数据点
      the_first = temp[((temp['Date'] >= start1) & (temp['Date'] <= end3)) | (temp['Date'] >= start_e)]['LastPrice'].values[0]
      temp = temp.shift(1)
      temp.drop(0, inplace=True)
      #print(temp)
      data = temp.resample('1min', on='Date', label='right', closed='right').agg(
              {'LastPrice': ['first', 'max', 'min', 'last'],
               'Volume': 'last', 'Turnover': 'last', 'OpenInterest': 'last',
               'BidPrice1': 'last', 'AskPrice1': 'last', 'AveragePrice': 'last',
               'UpperLimitPrice': 'last', 'LowerLimitPrice': 'last'})
      exchange_time_dict = exchange_time(day)
      #print(exchange_time_dict)
      #print(data)
      data['Date'] = data.index
      data = data.dropna(axis=0, how='any')
      data_type = 0
      #print(data,22222222)
    else:
      res = pd.DataFrame([], columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                        'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                        'LowerLimitPrice', 'Date', 'Avg_price'])
      return res
    #print(data)
    if exchange_time_dict is None:
        data_type = 1
        res = data[((data.index > start1) & (data.index <= end1)) | ((data.index > start2) & (data.index <= end2)) | ((data.index > start3) & (data.index <= end3))].copy()

    elif exchange_time_dict['hour'] > 21:
        data_type = 2
        end_e = pd.Timestamp(year=everning_day.year, month=everning_day.month, day=everning_day.day,
                                 hour=exchange_time_dict['hour'], minute=exchange_time_dict['min'], second=0)
        res = data[((data.index > start1) & (data.index <= end1)) | ((data.index > start2) & (data.index <= end2)) | ((data.index > start3) & (data.index <= end3)) | ((data.index > start_e) & (data.index <= end_e))].copy()
        
    elif exchange_time_dict['hour'] < 9:
        data_type = 3
        end_e = pd.Timestamp(year=everning_day.year, month=everning_day.month, day=everning_day.day,
                                 hour=0, minute=0, second=0) + pd.Timedelta(days=1)

        start4 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                                   hour=0, minute=0,second=0)

        end4 = pd.Timestamp(year=day.year[0], month=day.month[0], day=day.day[0],
                                 hour=exchange_time_dict['hour'], minute=exchange_time_dict['min'],
                                 second=0)

        res = data[((data.index > start1) & (data.index <= end1)) | ((data.index > start2) & (data.index <= end2)) | ((data.index > start3) & (data.index <= end3)) | ((data.index > start_e) & (data.index <= end_e)) | ((data.index > start4) & (data.index <= end4))].copy()
    
    res['Date'] = res.index
    # 有夜盘数据
    # print(res)
    if (data_type ==2 or data_type == 3) and len(res[((res.index > start_e) & (res.index <= end_e))]) > 0:
        everning = res[((res.index > start_e) & (res.index <= end_e))]
        morning = res[((res.index > start1) & (res.index <= end1))]
        
        # 所以触发的else是有夜盘但没日盘，那就是垃圾数据了
        if len(everning) >0 and len(morning) >0:
            everning_day = everning['Date'][0].date()
            morning_day = morning['Date'][0].date()
        else:
            res = pd.DataFrame([], columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                            'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                            'LowerLimitPrice', 'Date', 'Avg_price'])
            return res
        if (everning_day - morning_day) == timedelta(days=0):
            diff = last_exchange_day(everning_day) - everning_day
            res.loc[((res.index > start_e) & (res.index <= end_e)),'Date'] = res.loc[((res.index > start_e) & (res.index <= end_e)), 'Date'].apply(lambda x: x+diff)
        
        res.reset_index(drop=True, inplace=True)
        # 必须要用稳定排序，因为有些tick收到的毫秒都显示为0,其实是有顺序的
        res.sort_values('Date', inplace=True, kind='mergesort')
        res.reset_index(drop=True, inplace=True)
        
        first_Turnover, first_volume, = res.loc[res.index[0], ('Turnover', 'last')],\
                                                res.loc[res.index[0], ('Volume', 'last')]

        res.loc[:, ('Turnover', 'last')] = res.loc[:, ('Turnover', 'last')].diff(1)
        res.loc[:, ('Volume', 'last')] = res['Volume']['last'].diff(1)
        res.loc[res.index[0], ('LastPrice', 'first')] = the_first
        res.loc[res.index[0], ('LastPrice', 'max')] = max(the_first, res.loc[res.index[0], ('LastPrice', 'max')])
        res.loc[res.index[0], ('LastPrice', 'min')] = min(the_first, res.loc[res.index[0], ('LastPrice', 'min')])

        res.loc[res.index[0], ('Turnover', 'last')] = first_Turnover
        res.loc[res.index[0], ('Volume', 'last')] = first_volume
        res.loc[:, 'Avg_price'] = res['Turnover']['last']/(margin*res['Volume']['last'])

        res.columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                            'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                            'LowerLimitPrice', 'Date', 'Avg_price']
        try:
            morning_open = res.loc[res[res['Date'] >= start1].index[0],'LastPrice_open']
            everning_open = res.loc[res[res['Date'] >= start_e].index[0],'LastPrice_open']
        except:
            res = pd.DataFrame([], columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                            'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                            'LowerLimitPrice', 'Date', 'Avg_price'])
            return res

        res['LastPrice_open'] = res['LastPrice_close'].shift(1).values
        res.loc[res[res['Date'] >= start1].index[0],'LastPrice_open'] = morning_open
        res.loc[res[res['Date'] >= start_e].index[0],'LastPrice_open'] = everning_open
        #print(res)
    elif data_type == 1 or len(res[((res.index > start_e) & (res.index <= end_e))]) == 0:
        res.reset_index(drop=True, inplace=True)
        if len(res) <= 0:
          res = pd.DataFrame([], columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                        'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                        'LowerLimitPrice', 'Date', 'Avg_price'])
          return res
        first_Turnover, first_volume, = res.loc[res.index[0], ('Turnover', 'last')],\
                                                res.loc[res.index[0], ('Volume', 'last')]

        res.loc[:, ('Turnover', 'last')] = res.loc[:, ('Turnover', 'last')] .diff(1)
        res.loc[:, ('Volume', 'last')] = res['Volume']['last'].diff(1)
        res.loc[res.index[0], ('LastPrice', 'first')] = the_first
        res.loc[res.index[0], ('LastPrice', 'max')] = max(the_first, res.loc[res.index[0], ('LastPrice', 'max')])
        res.loc[res.index[0], ('LastPrice', 'min')] = min(the_first, res.loc[res.index[0], ('LastPrice', 'min')])

        res.loc[res.index[0], ('Turnover', 'last')] = first_Turnover
        res.loc[res.index[0], ('Volume', 'last')] = first_volume
        res.loc[:, 'Avg_price'] = res['Turnover']['last']/(margin*res['Volume']['last'])
        res.columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                            'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                            'LowerLimitPrice', 'Date', 'Avg_price']

        # 这里是只有日盘
        morning_open = res.loc[res[res['Date'] >= start1].index[0],'LastPrice_open']
        res['LastPrice_open'] = res['LastPrice_close'].shift(1).values
        res.loc[res[res['Date'] >= start1].index[0],'LastPrice_open'] = morning_open

    else:
        res = pd.DataFrame([], columns = ['LastPrice_open', 'LastPrice_high', 'LastPrice_low', 'LastPrice_close', 'Volume', 'Turnover',
                        'OpenInterest_last', 'BidPrice1', 'AskPrice1', 'AveragePrice', 'UpperLimitPrice',
                        'LowerLimitPrice', 'Date', 'Avg_price'])

    # print(res)
    return res


def last_exchange_day(calendar):
    def res(today):
        pos = calendar[calendar['trade_date'] == today].index[0]
        last_day = calendar.loc[pos - 1, 'trade_date']
        return last_day
    return res


def get_filename(our_path: str, contract: str):
    os.chdir(path=our_path+contract)
    now_path = os.getcwd()
    data_name = os.listdir(now_path)
    contract_dict = defaultdict(list)
    for item in data_name:
        name = re.findall(r'(\S+)_', item)[0]
        contract_dict[name].append(item)

    return contract_dict


def save_all(data_path: str, save_path: str, contract: str, margin: int, exchange_time, get_last_day):
    contract_dict = get_filename(our_path=data_path, contract=contract)
    length = len(contract_dict.keys())
    
    save_path_list = [save_path + contract] * length
    margin_list = [margin] * length
    exchange_time_list = [exchange_time] * length
    get_last_day_list = [get_last_day] * length

    for key, value in contract_dict.items():
        get_data(save_path+contract, key, value, margin, exchange_time, get_last_day)

    #res = list(map(get_data, save_path_list, list(contract_dict.keys()), list(contract_dict.values()), margin_list, exchange_time_list, get_last_day_list))


def sort_data(my_path: str):
    os.chdir(path=my_path)
    now_path = os.getcwd()
    data_name = os.listdir(now_path)
    # print(data_name)
    for item in data_name:
        if data_name[:-3] != 'csv':
          continue
        data = pd.read_csv(item)
        data.sort_values('Date', inplace=True, kind='mergesort')
        data.to_csv(item, index=False)


def exchange_time_function(contract):
    r = pd.read_csv('/mnt/DataServer/share/future_datas_zc/exchange_time.csv')
    exchange = eval(r[r['code'] == contract]['everning_end'].values[0])
    exchange = {pd.to_datetime(k): v for k, v in exchange.items()}
    for k, v in exchange.items():
        for kk, vv in v.items():
            v[kk] = int(vv)
    exchange_t = sorted(exchange.items(),key=lambda d:d[0])
    exchange2 = {}
    for item in exchange_t:
        exchange2[item[0]] = item[1]

    def exchange_times(day):
        if not exchange2:
            return None
        all_day = list(exchange2.keys())
        for n, key in enumerate(all_day):
            if day < key and n == 0:
                return None
            if day < key and n > 0:
                return exchange2[all_day[n-1]]
        return exchange2[all_day[-1]]

    return exchange_times

def run_get_data_plus1():
    path = '/mnt/DataServer/share/future_datas_zc/group_by_type/'
    out_put_path = '/mnt/DataServer/share/future_datas_zc/datas_1min/'
    
    exchange = pd.read_csv('/mnt/DataServer/share/future_datas_zc/exchange_time.csv')
    margin_csv = pd.read_csv('/mnt/DataServer/Data/future/margin.csv')
    calendar = pd.read_csv('/mnt/DataServer/Data/future/calendar.csv')
    calendar['trade_date'] = calendar['trade_date'].apply(lambda x:pd.to_datetime(x).date())
    get_last_day = last_exchange_day(calendar=calendar)

    os.chdir(path=path)
    # now_path = os.getcwd()
    # data_name = os.listdir(now_path)
    #data_name = ['hc', 'rb', 'ru', 'y'] ##0
    #data_name = ['m', 'j', 'i', 'p'] ##1
    #data_name = ['c', 'cs','ag', 'au'] ##2
    #data_name = ['l', 'v','jm'] ##3
    #data_name = ['bu'] ##4
    #data_name = ['p']
    #data_name = ['TA', 'MA', 'FG', 'CF']
    # data_name = ['AP', 'C', 'CF', 'FG', 'HC', 'I', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SF', 'SM', 'SR', 'V', 'Y', 'ZC']
    # data_name = ['MA', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SF', 'SM', 'SR', 'V', 'Y', 'ZC', 'M']
    # data_name = ['ag', 'au', 'cu', 'al', 'zn', 'pb', 'ni', 'sn', 'sc', 'bu']
    data_name = ['AP', 'FG', 'HC', 'JD', 'JM', 'L', 'M', 'OI', 'P', 'RM', 'RU', 'V', 'SN', 'pp']
    for item in data_name:
    #   if item in ['CF']:
    #      continue
      print(item)
      if not os.path.exists(out_put_path+item):
          os.mkdir(out_put_path+item)

      if item.upper() in exchange['code'].values:
          exchange_time = exchange_time_function(item.upper())
          multiplier = margin_csv[margin_csv['code'] == item.upper()]['multiplier'].values[0]
          save_all(data_path=path, save_path=out_put_path, contract=item, exchange_time=exchange_time, margin=multiplier, get_last_day=get_last_day)
          sort_data(my_path=out_put_path+item+'/')
      else:
          print(item + '未找到此code')


if __name__ == '__main__':
    # path = '/mnt/zpool1/Data/future/group_by_type/'
    # out_put_path = '/mnt/zpool1/Data/future/data_1min/'
    # path = '/mnt/DataServer/Data/future/group_by_type/'
    # run_group_by_type()
    # run_qh_marketdata()
    path = '/mnt/DataServer/share/future_datas_zc/group_by_type/'
    out_put_path = '/mnt/DataServer/share/future_datas_zc/datas_1min/'
    
    exchange = pd.read_csv('/mnt/DataServer/share/future_datas_zc/exchange_time.csv')
    margin_csv = pd.read_csv('/mnt/DataServer/Data/future/margin.csv')
    calendar = pd.read_csv('/mnt/DataServer/Data/future/calendar.csv')
    calendar['trade_date'] = calendar['trade_date'].apply(lambda x:pd.to_datetime(x).date())
    get_last_day = last_exchange_day(calendar=calendar)

    os.chdir(path=path)
    # now_path = os.getcwd()
    # data_name = os.listdir(now_path)
    #data_name = ['hc', 'rb', 'ru', 'y'] ##0
    #data_name = ['m', 'j', 'i', 'p'] ##1
    #data_name = ['c', 'cs','ag', 'au'] ##2
    #data_name = ['l', 'v','jm'] ##3
    #data_name = ['bu'] ##4
    #data_name = ['p']
    #data_name = ['TA', 'MA', 'FG', 'CF']
    # data_name = ['AP', 'C', 'CF', 'FG', 'HC', 'I', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SF', 'SM', 'SR', 'V', 'Y', 'ZC']
    # data_name = ['MA', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SF', 'SM', 'SR', 'V', 'Y', 'ZC', 'M']
    # data_name = ['ag', 'au', 'cu', 'al', 'zn', 'pb', 'ni', 'sn', 'sc', 'bu']
    data_name = ['JM', 'HC', 'L', 'M', 'OI', 'P', 'RM', 'RU', 'V', 'SN', 'pp']
    for item in data_name:
    #   if item in ['CF']:
    #      continue
        print(item)
        if not os.path.exists(out_put_path+item):
            os.mkdir(out_put_path+item)
        if item.upper() in exchange['code'].values:
            exchange_time = exchange_time_function(item.upper())
            multiplier = margin_csv[margin_csv['code'] == item.upper()]['multiplier'].values[0]
            save_all(data_path=path, save_path=out_put_path, contract=item, exchange_time=exchange_time, margin=multiplier, get_last_day=get_last_day)
            sort_data(my_path=out_put_path+item+'/')
        else:
            print(item + '未找到此code')