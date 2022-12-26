import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import warnings
warnings.filterwarnings("ignore")

from data_process import *
from data_load import RawData


class MainCon:
    '''
    get_raw_maincon_data: 获取所有合约数据（标识主力、次主力）
    get_raw_vp_data     : 获取主力量价数据
    get_adj_vp_data     : 获取主力连续量价数据
    '''
    def __init__(self) -> None:
        self.path = f'{pa_prefix}/datas/mainconinfo/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    def change_symbol(self, df: DataFrame):
        '''郑商所代码为3位，补齐第一位，如AP001，可能为2010年或2020年的合约'''
        if len(list(filter(str.isdigit, df['symbol'].values[0]))) == 3:
            df['symbol_chg'] = np.nan
            for i in df.index:
                for num in ['1', '2']:
                    symbol_raw = list(filter(str.isdigit, df.loc[i, 'symbol']))     # 001
                    symbol_new = ''.join([num]+symbol_raw)[:2]                      # 20
                    date = df.loc[i, 'date'].strftime("%Y-%m-%d").split('-')[0][-2:]    # 19
                    if int(date)-5<int(symbol_new)<int(date)+5:     # 14<20<24
                        df.loc[i, 'symbol_chg'] = ''.join(symbol_new+''.join(symbol_raw)[1:])   # 2001
        else:
            df['symbol_chg'] = np.nan
            for i in df.index:
                symbol = list(filter(str.isdigit, df.loc[i, 'symbol']))
                df.loc[i, 'symbol_chg'] = ''.join(symbol)
        return df

    def change_maincon_to_rq(self):
        '''将主力合约表转成标准表'''
        df = pd.read_csv(f'{self.path}df_symbol_chg_maincon.csv', index_col=0)
        df = df[df['maincon']==True] 
        df['symbol'] = df['object'] + df['symbol_chg'].apply(str)
        df = df[['date', 'symbol', 'object']]
        df.columns = ['date', 'contract', 'symbol']
        df['contract'] = df['contract'].apply(lambda x: x.upper())
        df['contract1'] = df['contract'].shift(1)
        df = df.iloc[1:]
        df['is_change'] = np.where(df['contract']==df['contract1'], 0, 1)
        df = df[['date', 'contract', 'contract1', 'is_change', 'symbol']]
        df.to_csv(f'{pa_prefix}/datas/maincon_my.csv', index=False)

    def select_maincon(self, df: DataFrame):
        '''标识主力'''
        old_maincon = 0      # 上一个交易日的主力
        new_maincon = 0      # 当前交易日的主力
        df_all = pd.DataFrame()
        for index, i in enumerate(df['date'].unique()):

            df_i = df[df['date']==i].copy()
            # 成交量、持仓量最大的某行（空 或 一行）
            vi_max = df_i[(df_i['to_volume'] == df_i['to_volume'].max()) & (df_i['interest']==df_i['interest'].max())]
            if vi_max.shape[0] > 2:
                print('vi_max行数大于1行')
            
            # 如果成交量持仓量最大为同一个合约
            if not vi_max.empty:

                new_maincon = vi_max['symbol_chg'].values[0]    # 更新新主力
                # print(new_maincon, old_maincon)
                # print('-----------')
                # 判断新主力合约是否比旧主力合约小 或 涨跌停
                if new_maincon < old_maincon \
                    or (vi_max['close'].values == vi_max['LIMIT_UP_PRICE'].values \
                        and vi_max['close'].values == vi_max['LIMIT_DOWN_PRICE'].values):
                    # 如果old_maincon未到期
                    if old_maincon in list(df_i['symbol_chg']):
                        new_maincon = old_maincon       # 新主力重置为旧主力
                
                df_i['maincon'] = (df_i['symbol_chg'] == new_maincon)     # 新增列'maincon'
                
                old_maincon = new_maincon           # 更新下交易日的旧主力
            
            # 成交量、持仓量最大为不同合约
            else:

                vmax = df_i[df_i['to_volume'] == df_i['to_volume'].max()]
                imax = df_i[df_i['interest'] == df_i['interest'].max()]
                # 选取最大成交量、最大持仓量中合约名称最小的
                new_maincon = min(vmax['symbol_chg'].values[0], imax['symbol_chg'].values[0])

                # 如果新合约成交量小于100（防止频繁更换） 或 合约名称小 或 涨跌停，则不更换新合约
                df_new = df_i[df_i['symbol_chg']==new_maincon]
                if df_new['to_volume'].values[0] < 100 \
                    or new_maincon < old_maincon \
                    or (df_new['close'].values == df_new['LIMIT_UP_PRICE'].values \
                        and df_new['close'].values == df_new['LIMIT_DOWN_PRICE'].values):
                    # 如果old_maincon未到期
                    if old_maincon in list(df_i['symbol_chg']):
                        new_maincon = old_maincon
                
                df_i['maincon'] = (df_i['symbol_chg'] == new_maincon)     # 新增列'maincon'

                old_maincon = new_maincon           # 更新下交易日的旧主力

            df_all = pd.concat([df_all, df_i], axis=0)

            # # debug
            # old = df_all[df_all['date']==df_all['date'].unique()[index-1]]
            # new = df_all[df_all['date']==df_all['date'].unique()[index]]
            # oldm = old.loc[old['maincon']==True, 'symbol_chg'].values[0]
            # newm = new.loc[new['maincon']==True, 'symbol_chg'].values[0]
            # if index>0 and oldm>newm:
            #     print(df_i['symbol_chg'])
            #     input()
        return df_all

    def select_smaincon(self, df: DataFrame):
        '''标识次主力'''
        old_smaincon = 0      # 上一个交易日的次主力
        new_smaincon = 0      # 当前交易日的次主力
        df_all = pd.DataFrame()

        for index, i in enumerate(df['date'].unique()):
            df_i = df[df['date']==i].copy()
            maincon = df_i.loc[df_i['maincon']==True, 'symbol_chg'].values[0]
            df_s = df_i[df_i['symbol_chg'] != maincon]     # 选出非主力合约的行
            # 成交量、持仓量次最大的某行（空 或 一行）
            vi_max = df_s[(df_s['to_volume'] == df_s['to_volume'].max()) & (df_s['interest']==df_s['interest'].max())]

            # 无成交量
            if vi_max.shape[0] >= 2:
                # print('vi_max行数大于1行且成交量为0')
                # print(vi_max)
                df_i['smaincon'] = np.nan
            
            # 如果成交量持仓量次最大为同一个合约
            if vi_max.shape[0] == 1:
                new_smaincon = vi_max['symbol_chg'].values[0]    # 更新新次主力

                # 判断新次主力合约是否比旧次主力合约小 或 涨跌停
                if new_smaincon < old_smaincon:
                    # 如果old_maincon未到期
                    if old_smaincon in list(df_s['symbol_chg']):
                        new_smaincon = old_smaincon       # 新主力重置为旧主力

                df_i['smaincon'] = (df_s['symbol_chg'] == new_smaincon)     # 新增列'smaincon'
                df_i['smaincon'].fillna(False, inplace=True)

                old_smaincon = new_smaincon           # 更新下交易日的旧主力

            elif vi_max.shape[0] == 0:
                vmax = df_s[df_s['to_volume'] == df_s['to_volume'].max()]
                imax = df_s[df_s['interest'] == df_s['interest'].max()]
                
                # 如果较小合约小于主力，则次主力为较大合约
                if min(vmax['symbol_chg'].values[0], imax['symbol_chg'].values[0]) < maincon:
                    new_smaincon = max(vmax['symbol_chg'].values[0], imax['symbol_chg'].values[0])
                # 如果较小合约大于主力，则次主力为较小合约
                else:
                    new_smaincon = min(vmax['symbol_chg'].values[0], imax['symbol_chg'].values[0])

                # 如果新合约成交量小于100（防止频繁更换） 且 旧次主力不是主力 且 old_maincon未到期
                if df_s.loc[df_s['symbol_chg']==new_smaincon, 'to_volume'].values[0] < 100 \
                    and old_smaincon != maincon \
                    and old_smaincon in list(df_s['symbol_chg']):

                    new_smaincon = old_smaincon       # 新主力重置为旧主力

                df_i['smaincon'] = (df_s['symbol_chg'] == new_smaincon)     # 新增列'smaincon'
                df_i['smaincon'].fillna(False, inplace=True)

                old_smaincon = new_smaincon           # 更新下交易日的旧主力
            
            # 会有字符串的true？？
            if df_i[df_i['smaincon']=='True'].shape[0] != 0:
                df_i.loc[df_i['smaincon']=='True', 'smaincon'] = True

            df_all = pd.concat([df_all, df_i], axis=0)
        return df_all

    def get_raw_maincon_data(self):
        '''获取所有合约数据（标识主力、次主力）'''
        if not os.path.exists(self.path+'df_symbol_chg.csv'):
            print('正在修改标的代码...')
            data = RawData().query_raw_volprice_data_myrule().copy()
            data = data.groupby(['object'],as_index=False).apply(self.change_symbol)
            data.to_csv(self.path+'df_symbol_chg.csv')
            print('标的代码修改完成')
        else:
            data = pd.read_csv(self.path+'df_symbol_chg.csv', index_col=0)

        if not os.path.exists(self.path+'df_symbol_chg_maincon.csv'):
            print('正在选择主力合约...')
            data = data.groupby(['object'],as_index=False).apply(self.select_maincon)
            data.index = data.index.droplevel(0)
            data.to_csv(self.path+'df_symbol_chg_maincon.csv')
            print('主力合约选择完成')
        else:
            data = pd.read_csv(self.path+'df_symbol_chg_maincon.csv', index_col=0)

        if not os.path.exists(self.path+'df_symbol_chg_maincon_s.csv'):
            print('正在选择次主力合约...')
            data = data.groupby(['object'],as_index=False).apply(self.select_smaincon)
            data.index = data.index.droplevel(0)
            data.to_csv(self.path+'df_symbol_chg_maincon_s.csv')
            print('次主力合约选择完成')
        else:
            data = pd.read_csv(self.path+'df_symbol_chg_maincon_s.csv', index_col=0)
        
        data.drop(['LIMIT_UP_PRICE','LIMIT_DOWN_PRICE', 'symbol_chg'], axis=1, inplace=True)
        data = data[~data['smaincon'].isin([np.nan])]   # 去除成交量为0的行
        return data

    def get_raw_vp_data(self):
        '''获取主力量价数据'''
        if not os.path.exists(self.path+'raw_vp_data.csv'):
            data  = self.get_raw_maincon_data()
            data = data[data['maincon']==True]
            data.drop(['last_deli_date', 'datediff', 'maincon', 'smaincon'], axis=1, inplace=True)
            data.to_csv(self.path+'raw_vp_data.csv')
        else:
            data = pd.read_csv(self.path+'raw_vp_data.csv', index_col=0)
        return data

    

    '''复权'''

    def adjust_maincon(self, df: DataFrame, type='forward'):
        '''
        某只标的的主力连续合约
                                    新主力合约换月前收盘价
        复权因子T-1 = 复权因子T *  ——————————————————————————
                                    旧主力合约换月前收盘价

        type: 'forward':前复权 / 'back':后复权
        '''
        def save(df, save=False, show=False):
            data = df.copy()
            name = data['object'].values[0]
            data['close_adj'] = data['close']*data['adj_multiplier']      # 前复权
            data['open_adj'] = data['open']*data['adj_multiplier']        # 前复权
            data['high_adj'] = data['high']*data['adj_multiplier']        # 前复权
            data['low_adj'] = data['low']*data['adj_multiplier']          # 前复权
            if save:
                if not os.path.exists('data/forward_adjust'):
                    os.makedirs('data/forward_adjust')
                data.to_csv('data/forward_adjust/{}_adj.csv'.format(name))
            if show:
                x = data['date']
                y1 = data['close']
                y2 = data['close_adj']
                plt.plot(x, y1)
                plt.plot(x, y2)
                # debug(df)
                print(data)
                plt.show()

        df['is_chg'] = df['symbol'] != df['symbol'].shift(1)    # 是否换合约
        df['is_chg'][df.index[0]] = False   # 第一行换成False

        df['fixed_factor'] = (df['close'].shift(1)/df['pre_close'])     # 初始固定复权因子

        # 选出is_chg为False的行，将fixed_factor置为空值
        df.loc[df['is_chg']==False, 'fixed_factor'] = np.nan
        # 选出is_chg为True的行，将fixed_factor置为累积
        df.loc[df['is_chg']==True, 'fixed_factor'] = np.cumprod(df.loc[df['is_chg']==True, 'fixed_factor'])

        df['fixed_factor'] = df['fixed_factor'].fillna(method='ffill').fillna(1)    # 向下填充

        # update_factor为每次更新数据时需要更新的复权因子,若为1，等价于后复权因子
        if type == 'forward':
            df['update_factor'] = df['fixed_factor'].values[-1]      
        elif type == 'back':
            df['update_factor'] = 1
        else:
            print('复权方式type错误')

        df['adj_multiplier'] = df['fixed_factor']/df['update_factor']    # 最终复权系数：固定复权因子/换月时更新的复权因子

        save(df, save=True, show=False)

        df['close'] = df['close']*df['adj_multiplier']          # 前复权
        df['open'] = df['open']*df['adj_multiplier']            # 前复权
        df['high'] = df['high']*df['adj_multiplier']            # 前复权
        df['low'] = df['low']*df['adj_multiplier']              # 前复权
        df['settle'] = df['settle']*df['adj_multiplier']        # 前复权

        del df['symbol'], df['pre_close'], df['is_chg'], df['fixed_factor'], df['update_factor'], df['adj_multiplier']

        return df

    def get_adj_vp_data(self):
        '''获取主力连续量价数据'''
        if not os.path.exists(self.path+'adj_vp_data.csv'):
            data = self.get_raw_vp_data()
            # 根据self.contract_list中的合约计算各自主连数据
            data = data.groupby(['object'],as_index=False).apply(self.adjust_maincon)
            # df.index = df.index.droplevel(0)
            data.to_csv(self.path+'adj_vp_data.csv')
        else:
            data = pd.read_csv(self.path+'adj_vp_data.csv', index_col=0)
        return data
    


    '''debug'''

    def debug1(self):
        '''找出更新合约比旧合约小的情况'''
        def debug(df):
            s = df['symbol_chg']-df['symbol_chg'].shift(1)
            if s[s<0].shape[0] != 0:
                print('存在更新合约比旧合约小的情况')
                print(df.loc[s<0,:])
                input()
            return df
        df = self.get_raw_vp_data()
        df.groupby(['object'],as_index=False).apply(debug)

    def debug2(self):
        '''找出某品种当日主力不唯一的情况'''
        def debug(df):
            if df[df['maincon']==True].shape[0] != 1:
                print('存在某品种当日主力不唯一的情况')
                print(df)
                input()
            if df[df['smaincon']==True].shape[0] != 1:
                print('存在某品种当日次主力不唯一的情况')
                print(df)
                print(df.loc[df.index[0], 'smaincon'])
                print(df.loc[df.index[0], 'smaincon']==True)
                print(df.loc[df.index[0], 'smaincon']=='True')
                print(df[df['smaincon']==True])
                input()
            if df[(df['maincon']==True) & (df['smaincon']==True)].shape[0] != 0:
                print('存在主力次主力相同的情况')
                print(df)
                input()
            return df
        df = self.get_raw_maincon_data()
        df.groupby(['object', 'date'],as_index=False).apply(debug)

    def debug3(self):
        '''换月频率'''
        def debug(df: DataFrame):
            df = df.copy()
            df['is_chg'] = df['symbol'] != df['symbol'].shift(1)    # 是否换合约
            df['is_chg'][df.index[0]] = False   # 第一行换成False
            df.reset_index(drop=True, inplace=True)
            df = df[df['is_chg']==True]
            index = df.index.values
            days = []
            for i in range(len(index)-1):
                d = index[i+1] - index[i]
                days.append(d)
                if d < 5:
                    print(df.loc[index[i-1]:index[i+1], :])
            print(days)
            input()
            return df
        df = self.get_raw_maincon_data()
        df = df[df['maincon']==True]
        df.groupby(['object'],as_index=False).apply(debug)

def run_MainCon():
    maincon = MainCon()
    maincon.get_raw_maincon_data()
    # maincon.change_maincon_to_rq()

def run_change_maincon_to_rq():
    try:
        run_MainCon()
    except:
        pass
    run_MainCon()
    maincon = MainCon()
    maincon.change_maincon_to_rq()


if __name__ == "__main__":
    # run_MainCon()
    run_change_maincon_to_rq()
