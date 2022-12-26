#%%
import sys, os
from traceback import print_last
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
from datas_process.zigzag import ZigZagInfo, Zigzag
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
# from backtesting.ml_strategy import MLStrategy
from backtesting.ml_strategy_adj import MLStrategy
from backtesting.simulation_strategy import SimulationStrategy
from datetime import datetime, time
from datas_process.m_futures_factors import SymbolsInfo, MainconInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting.model_statistics import ModelStatistics
from datas_process.m_datas_process import BaseDataProcess
# from vnpy_portfoliostrategy import BacktestingEngine
# from vnpy.trader.constant import Interval
# from atr_rsi_strategy import AtrRsiStrategy
from functools import partial
# import swifter
from m_base import Logger, get_sy, timestamp_to_datetime, train_val_test_pnl_plot
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
sys.path.append('..')
# sys.stdout = Logger('./datas/backtest_res/log.txt')

__Author__ = 'ZCXY'

class BackTester():
    def __init__(self, startdate=datetime(2016, 1, 1), enddate=datetime(2020, 11, 1)):  # 
        self.startdate = startdate
        self.enddate = enddate
        self.sig_meth = 0  # 0预测 1概率预测 2真实 3二分类
        self.res_pa = f'{pa_prefix}/datas/backtest_res/'
        self.pred_pa = f'{pa_prefix}/datas/predict/'
        self.syinfo = SymbolsInfo()
        self.mainconinfo = MainconInfo()
        self.contract_rate = self.syinfo.contract_rate
        self.df_symbols_all = self.syinfo.df_symbols_all
        self.capital = 1_000_000

        if not os.path.exists(self.res_pa):
            os.makedirs(self.res_pa)

    def get_backtest_params(self, symbol):
        '''获取回测合约参数'''
        contract = self.contract_rate[self.contract_rate[' 合约代码']==symbol.upper()]
        symbol_info = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]
        price = symbol_info['price'].iloc[0]
        pricetick = symbol_info['pricetick'].iloc[0]
        size = symbol_info['size'].iloc[0]
        cr1 = contract[' 开仓手续费(按手数)'].iloc[0]
        cr2 = contract[' 开仓手续费(按金额)'].iloc[0]
        rate = cr2 if cr1 == 0 else cr1 / (price*size)
        hand = int(300_000 / (price*size*0.15))
        hand = hand if hand > 0 else 1
        if hand <= 0:
            print('symbol', symbol, 'hand', hand)
            input()
        return rate, pricetick, size, hand

    def backtesting(self, contract, startdate, enddate, y_pred, plot=True, need_engine=0):
        '''跑回测'''
        symbol = get_sy(contract)
        rate, pricetick, size, hand = self.get_backtest_params(symbol)
        startdate = self.startdate if self.startdate > startdate else startdate
        enddate = self.enddate if self.enddate < enddate else enddate
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=f"{contract}.LOCAL",
            interval="1m",
            start=startdate,
            end=enddate,
            rate=rate,
            slippage=pricetick,
            size=size,
            pricetick=pricetick,
            capital=1_000_000,
        )
        params = {'hand': hand, 'symbol_name': symbol, 'contract': contract, 'y_pred': y_pred,    # 'y_pred_[10, 1, 1, 0]_ru_5m_0.6_50_1_return_rate_60m'
                    'sig_meth': self.sig_meth, 'step_n': 1, 'win_n': 1, 'rate': rate, 'size': size, 'pricetick': pricetick}
                    
        engine.add_strategy(SimulationStrategy, params)
        # engine.add_strategy(AtrRsiStrategy, params)

        engine.load_data()
        engine.run_backtesting()
        df = engine.calculate_result()
        res = engine.calculate_statistics(output=True)
        
        if plot:
            engine.show_chart()

        res.update({'symbol': symbol})
        
        # if pa == None:
        #     pa = self.res_pa
        # ax = pd.DataFrame(df["balance"]).plot()
        # fig = ax.get_figure()
        # fig.savefig(pa+symbol+str(self.sig_meth)+'.png')
        # plt.close()

        # setting = OptimizationSetting()
        # setting.set_target("sharpe_ratio")
        # setting.add_parameter("atr_length", 25, 27, 1)
        # setting.add_parameter("atr_ma_length", 10, 30, 10)
        # engine.run_ga_optimization(setting)
        # engine.run_bf_optimization(setting)
        if need_engine:
            if res['sharpe_ratio'] != 0:
                return engine, res
            else:
                return 0, 0
        else:
            del engine
            return res, df
    
    def get_target(self, target_type='drawdown', df_res=None):
        '''获取目标值'''
        try:
            if np.sum(df_res['signal'].abs()) < 20:
                return -3
        except:
            pass

        if target_type == 'drawdown':
            target = -self.caculate_drawdown(df_res)
            if target == 0 and df_res['pnl_cost'].iloc[-1] == 0:
                print('target为0', target)
                target = -3
        elif target_type == 'total_profit':
            target = self.caculate_total_profit(df_res)
        
        elif target_type == 'max_ddpercent':
            target = self.caculate_max_ddpercent(df_res)
        
        elif target_type == 'profit_rate_drawdown':
            target = self.caculate_profit_rate_drawdown(df_res)
        return target

    def caculate_drawdown(self, df_res: pd.DataFrame):
        '''计算最长回撤周期
        df: datetime, close, profit, cost, pnl, pnl_cost
        '''
        df = df_res.copy()
        df['highlevel'] = (df["pnl_cost"].rolling(min_periods=1, window=len(df), center=False).max())
        df_i = df.drop_duplicates(subset=['highlevel'], keep='first')
        df_i['ind'] = df_i.index.to_list()
        if len(df_i)>3:
            target = (df_i['ind'] - df_i['ind'].shift(1)).max()
            target = max(target, len(df)-df_i['ind'].iloc[-1]) / len(df)
        else:
            print('len_df_i:', len(df_i))
            target = 1
        # df["drawdown"] = df["pnl_cost"] - df["highlevel"]
        # max_drawdown_end = df["drawdown"].idxmin()
        # max_drawdown_start = df["pnl_cost"][:max_drawdown_end].idxmax()
        # target = max_drawdown_end - max_drawdown_start
        return target

    def caculate_sharp_ratio(self, df: pd.DataFrame):
        '''计算夏普比率'''
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        # res = df_res['profit'].groupby(df_res['date']).apply(np.sum)
        
        res = df['profit_cost'].groupby(df['date']).apply(np.sum)
        target = res.mean() / res.std() * np.sqrt(244)
        return target

    def caculate_total_profit(self, df):
        '''总收益率'''
        return df['pnl_cost'].iloc[-1]

    def caculate_max_ddpercent(self, df):
        '''计算百分比最大回撤'''
        df["highlevel"] = (
            df["pnl_cost"].rolling(
                min_periods=1, window=len(df), center=False).max()
        )
        df["drawdown"] = df["pnl_cost"] - df["highlevel"]
        # df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100
        # max_ddpercent = df["ddpercent"].min()
        max_drawdown = df['drawdown'].min()
        return max_drawdown
    
    def caculate_max_ddpercent_duration(self, df):
        '''计算百分比最大回撤周期'''
        df["highlevel"] = (
            df["pnl_cost"].rolling(
                min_periods=1, window=len(df), center=False).max()
        )
        df["drawdown"] = df["pnl_cost"] - df["highlevel"]
        max_drawdown_end = df["drawdown"].idxmin()
        try:
            max_drawdown_start = df["pnl_cost"][:max_drawdown_end].idxmax()
        except:
            return -3

        max_drawdown_duration = -(max_drawdown_end - max_drawdown_start) / (244*6)
        return max_drawdown_duration

    def caculate_profit_rate_drawdown(self, df):
        '''收益回撤比'''
        max_drawdown = self.caculate_max_ddpercent(df)
        return abs((df["pnl_cost"].iloc[-1] - 1) / max_drawdown)

    def caculate_zigzag_return(self, symbol, df_res, cat):
        '''zigzag收益'''
        ms = ModelStatistics()
        try:
            df_res_concat = ms.caculate_statistics_trend_shock(symbol, df_res, save_pa=None)
        except:
            print('计算zigzag目标函数时出错！')
            return -3
        target_trend, target_shock = df_res_concat.loc['按比例实际年化收益']
        target = target_trend if cat == 'trend' else target_shock
        return target
        
    def all_contract_backtesting(self, symbol, startdate=None, enddate=None, y_pred=None, target_type='drawdown'):
        '''单品种全合约回测'''
        capital = 1_000_000
        if startdate==None:
            startdate, enddate = self.startdate, self.enddate
        df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, startdate, enddate, delay=2)  # 2 修改
        annual_return_li, res_li = [], []
        for i in range(df_contractinfo.shape[0]):
            q = df_contractinfo.iloc[i].to_list()
            try:
                engine, res = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), y_pred, plot=False, need_engine=1)
            except:
                engine = 0   # 修改
            if engine is 0:
                continue
            df_i = pd.DataFrame(engine.strategy.res_dic)
            # print('end: ', df_i['datetime'].iloc[-1])
            annual_return_li.append(df_i)
            res_li.append(res['total_return'])

        try:
            # target = np.sum(res_li)
            df_res = pd.concat(annual_return_li)
            df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
            df_res['profit_cost'] = df_res['profit'] - df_res['cost']
            df_res['pnl_cost'] = df_res['profit_cost'].cumsum() / capital + 1
            df_res.reset_index(drop=True, inplace=True)

            # target = self.caculate_sharp_ratio(df_res)
            if target_type == 'drawdown':
                target = -self.caculate_drawdown(df_res)
                if target == 0 and df_res['pnl_cost'].iloc[-1] == 0:
                    print('target为0', target)
                    target = -3
            elif target_type == 'total_profit':
                target = self.caculate_total_profit(df_res)
            elif target_type == 'zigzag_trend':
                target = self.caculate_zigzag_return(df_res, 'trend')
            elif target_type == 'zigzag_shock':
                target = self.caculate_zigzag_return(df_res, 'shock')
            del engine
        except:
            # print('出错了！！')
            target = -3
            df_res = 0

        return df_res, target
    
    def all_contract_backtesting_1(self, symbol, startdate=None, enddate=None, y_pred=None):
        '''单品种全合约回测'''
        if startdate==None:
            startdate, enddate = self.startdate, self.enddate
        df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, startdate, enddate, delay=0)
        balance_li, annual_return_li = [], []
        for i in range(df_contractinfo.shape[0]):
            q = df_contractinfo.iloc[i].to_list()
            print(q)
            # input()
            res, df = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), y_pred, plot=True)
            # df.to_csv('df.csv')
            # exit()
            # print(df.head(10))
            # print('1111111111111111111111111111')
            # print(df.tail(2))
            # print('2222222222222222222222222222')
            # input()
            if len(balance_li):
                df['balance'] += balance_li[-1]['balance'].iloc[-1] - 1_000_000
            balance_li.append(df)
            annual_return_li.append(res['sharpe_ratio'])
        try:
            df_all = pd.concat(balance_li)
            df_all = df_all[~df_all.index.duplicated(keep='first')]
            target = np.mean(annual_return_li)
        except:
            target = -10
            df_all = pd.DataFrame()

        # df_all.reset_index(inplace=True)
        df_all['balance'].plot()
        plt.show()
        return df_all, target

    def backtesting_all(self, pa=None):
        '''单进程回测所有品种'''
        res = []
        for symbol in self.df_symbols_all['symbol']:
            print('begin: ', symbol)
            res.append(self.backtesting(symbol))
            print('done: ', symbol)
        df_res = pd.DataFrame(res)
        if pa == None:
            pa = self.res_pa
        df_res.to_csv(pa+'res_all'+str(self.sig_meth)+'.csv')

    def multiprocess_backtesting():
        '''多进程跑回测'''
        pass
    
    def signal_analyze(self, symbol, pred_pa, traindate=None, valdate=None, testdate=None, enddate=None, save_df=1, params='tvt'):
        '''信号结果分析'''
        capital = 1_000_000

        # y_pred = './filter_results/ru/res2/y_pred_[5, 0.5, 1, 1]_ru_60m_0.6_50_1_return_rate_60m'
        # y_pred = './filter_results/v/res2/y_pred_[5, 0.5, 1, 0]_v_60m_0.6_50_1_return_rate_60m'
        if traindate is None:
            # traindate, valdate, testdate = datetime(2016, 1, 1), datetime(2019, 5, 1), datetime(2020, 5, 1)
            # enddate = datetime(2020, 10, 30)
            traindate, valdate, testdate = self.startdate, datetime(2019, 5, 1), datetime(2020, 5, 1)
            enddate = self.enddate
        
        save_pa=f'{pa_prefix}/datas/backtest_res/{symbol}/'
        makedir(save_pa)
        suffix = pred_pa.split('/')[-1][:-4]
        save_pa = f'{save_pa}{suffix}'

        # rate, pricetick, size, hand = self.get_backtest_params(symbol)  # 获取symbol相关属性

        if params == 'all':
            date_dic = {'all': [traindate, enddate]}
        elif params == 'tvt':
            date_dic = {'train': [traindate, valdate], 'val': [valdate, testdate], 'test': [testdate, enddate]}
        elif params == 'val':
            date_dic = {'val': [valdate, testdate]}

        data_li = []
        for i in date_dic:
            df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, date_dic[i][0], date_dic[i][1], delay=2)
            res = []
            # trade_res = []
            for j in range(df_contractinfo.shape[0]):
                q = df_contractinfo.iloc[j].to_list()
                print(q)
                # input()
                engine, _ = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), pred_pa, plot=0, need_engine=1)
                if engine is 0:
                    continue
                df_i = pd.DataFrame(engine.strategy.res_dic)
                res.append(df_i)
            #     for k in engine.trades.values():      # 获取交易记录
            #         trade_res.append([k.datetime, k.direction, k.offset, k.price])
            # trade_res = pd.DataFrame(trade_res)
            # trade_res.columns = ['datetime', 'direction', 'offset', 'price']
            # trade_res.to_csv(f'{save_pa}_{i}_trade.csv', index=False)
            df_res = pd.concat(res)
            df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
            df_res['pnl'] = df_res['profit'].cumsum() / capital + 1
            df_res['pnl_cost'] = (df_res['profit']-df_res['cost']).cumsum() / capital + 1
            data_li.append(df_res[['datetime', 'pnl_cost']].copy())
            # print(df_res[['datetime', 'pnl_cost']].copy())
            # print('len df_res:', len(df_res))

            if save_df:
                df_res.to_csv(f'{save_pa}_{i}_analyze.csv', index=False)
                # df_res.to_csv(f'analyze.csv', index=False)
        
        train_pa = f'{save_pa}_train_analyze'
        ms = ModelStatistics()  # 统计回测结果
        # ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_statistics', symbol=symbol)
        try:
            ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_statistics', symbol=symbol)
        except:
            print(train_pa, 'statistics got wrong.')
        ms.caculate_statistics_trend_shock_train_val_test(symbol, train_pa, f'{save_pa}_trend_shock_statistic')
        
        train_val_test_pnl_plot(data_li, save_pa)   # 将三个pnl曲线画在一张图并保存
        
        return df_res

    def signal_analyze1(self, symbol, pred_pa, load_pa, params='tvt'):
        '''信号结果分析'''
        capital = 1_000_000

        # y_pred = './filter_results/ru/res2/y_pred_[5, 0.5, 1, 1]_ru_60m_0.6_50_1_return_rate_60m'
        # y_pred = './filter_results/v/res2/y_pred_[5, 0.5, 1, 0]_v_60m_0.6_50_1_return_rate_60m'
        traindate, valdate, testdate = self.startdate, datetime(2019, 5, 1), datetime(2020, 5, 1)
        enddate = self.enddate
        
        save_pa=f'{pa_prefix}/datas/ml_result/model_2.0/max_drawdown_skew_skew_10_7/params/{symbol}/{load_pa}/'
        suffix = pred_pa.split('/')[-1][:-4]
        save_pa = f'{save_pa}{suffix}'

        # rate, pricetick, size, hand = self.get_backtest_params(symbol)  # 获取symbol相关属性

        if params == 'all':
            date_dic = {'all': [traindate, enddate]}
        elif params == 'tvt':
            date_dic = {'train': [traindate, valdate], 'val': [valdate, testdate], 'test': [testdate, enddate]}
        elif params == 'val':
            date_dic = {'val': [valdate, testdate]}

        data_li = []
        for i in date_dic:
            df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, date_dic[i][0], date_dic[i][1], delay=2)
            res = []
            # trade_res = []
            for j in range(df_contractinfo.shape[0]):
                q = df_contractinfo.iloc[j].to_list()
                print(q)
                # input()
                engine, _ = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), pred_pa, plot=0, need_engine=1)
                if engine is 0:
                    continue
                df_i = pd.DataFrame(engine.strategy.res_dic)
                res.append(df_i)
            #     for k in engine.trades.values():      # 获取交易记录
            #         trade_res.append([k.datetime, k.direction, k.offset, k.price])
            # trade_res = pd.DataFrame(trade_res)
            # trade_res.columns = ['datetime', 'direction', 'offset', 'price']
            # trade_res.to_csv(f'{save_pa}_{i}_trade.csv', index=False)
            df_res = pd.concat(res)
            df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
            df_res['pnl'] = df_res['profit'].cumsum() / capital + 1
            df_res['pnl_cost'] = (df_res['profit']-df_res['cost']).cumsum() / capital + 1
            data_li.append(df_res[['datetime', 'pnl_cost']].copy())
            # print(df_res[['datetime', 'pnl_cost']].copy())
            # print('len df_res:', len(df_res))

            # if save_df:
            #     df_res.to_csv(f'{save_pa}_{i}_analyze.csv', index=False)
                # df_res.to_csv(f'analyze.csv', index=False)
        
        train_pa = f'{save_pa}_train_analyze'
        ms = ModelStatistics()  # 统计回测结果
        # ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_statistics', symbol=symbol)
        try:
            ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_statistics', symbol=symbol)
        except:
            print(train_pa, 'statistics got wrong.')
        ms.caculate_statistics_trend_shock_train_val_test(symbol, train_pa, f'{save_pa}_trend_shock_statistic')
        
        # train_val_test_pnl_plot(data_li, save_pa)   # 将三个pnl曲线画在一张图并保存
        
        return None


class MyBackTester(BackTester):
    def __init__(self, startdate=datetime(2016, 1, 1), enddate=datetime(2020, 11, 1)):
        super().__init__(startdate, enddate)
        self.dp = BaseDataProcess()
            
    def set_mini_data(self, x, df):
        '''获取每小时前4分钟k线，用于开仓交易'''
        df = df[df['datetime']>=x]
        res_dic = {}
        # try:
        [res_dic.update({f'datetime{i}': [df['datetime'].iloc[i]], f'open{i}': [df['open'].iloc[i]],
                        f'high{i}': [df['high'].iloc[i]], f'low{i}': [df['low'].iloc[i]],
                        f'close{i}': [df['close'].iloc[i]]}) for i in range(4)]
        # except:
        #     print(x)
        #     print('----------')
        #     print(df.tail(20))
        #     input()
        #     [res_dic.update({f'datetime{i}': [None], f'open{i}': [None],
        #                     f'high{i}': [None], f'low{i}': [None],
        #                     f'close{i}': [None]}) for i in range(4)]
        df_res = pd.DataFrame(res_dic)
        return df_res.iloc[0]

    def backtesting(self, symbol, df_contract_li, y_pred, is_hand=1, open_method=1):
        '''
        跑回测, 添加了一分钟, 等市值
        y_pred: datetime y_pred
        '''
        balance = 1_000_000
        pricetick_type = 0
        # startdate = self.startdate if self.startdate > startdate else startdate
        # enddate = self.enddate if self.enddate < enddate else enddate
        rate, pricetick, size, hand = self.get_backtest_params(symbol)
        # hand = int(balance * hand * 0.15 / 300_000)
        real_pricetick = 0.00029 if pricetick_type else pricetick
        capital = 1_000_000
        pricetick_rate = 1  # 0.5
        hand = int(capital * hand * 0.15 / 300_000)
        res_li = []

        def get_open_price(x):
            '''计算开仓价格'''
            pre_close, pre_signal = x['pre_close'], x['pre_signal']
            if pre_signal == 0:
                return {'trade_time': [None], 'trade_price': [0]}
            price_li = [pre_close, x['close0'], x['close1'], x['close2'], x['close3']]
            trade_price, trade_time = 0, None
            
            dt, pre_dt = x['datetime'].time(), x['pre_datetime'].time()
            if open_method == 0:    # 1.0版本
                start_n = ((dt == time(9, 0) and (pre_dt == time(22, 0) or pre_dt == time(23, 0) or pre_dt == time(0, 0) or pre_dt == time(2, 0))) or \
                dt == time(13, 0) or dt == time(23, 0) or dt == time(0, 0) or dt == time(1, 0) or dt == time(3, 0))*1
            elif open_method == 1:  # 按照所有跳空第一分钟后再交易
                start_n = (dt == time(21, 0) or dt == time(13, 0) or dt == time(23, 0) or dt == time(9, 0))*1
            else:
                print('open_method got wrong, please check.')
                start_n = 0
                exit()
            # print(start_n, dt)
            # if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            # bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):

            for i in range(start_n, 4):
                if trade_price:
                    break
                else:
                    if pre_signal == -1 and price_li[i] <= x[f'high{i}']: 
                        trade_price, trade_time = max(x[f'open{i}']-pricetick, price_li[i]), x[f'datetime{i}']
                    elif pre_signal == 1 and price_li[i] >= x[f'low{i}']:
                        trade_price, trade_time = min(x[f'open{i}']+pricetick, price_li[i]), x[f'datetime{i}'] 
            
            if trade_price == 0:
                # print('trade_price is 0...', x)
                # input()
                trade_price, trade_time = price_li[-1], x['datetime3']
            
            return {'trade_time': [trade_time], 'trade_price': [trade_price]}

        for df_60 in df_contract_li:
            try:
                df_merge = pd.merge(df_60, y_pred, left_on='datetime', right_on='datetime', how='inner')
                df_merge['y_pred'] = df_merge['y_pred'] - 1 
                
                df_merge['pos'] = df_merge['y_pred'].replace(to_replace=0, value=None).shift(1)  # *df_merge['hand']
                df_merge['pos'].iloc[0] = 0     # 持仓

                if is_hand:
                    df_merge['hand'] = (capital / df_merge['close'] / size).apply(int).shift(1)
                    df_merge['hand'].iloc[0] = 0     # 持仓
                    df_merge['hand'] = np.where(df_merge['pos']!=df_merge['pos'].shift(1), df_merge['hand'], 0)
                    df_merge['hand'] = df_merge['hand'].replace(to_replace=0, value=None)
                else:
                    df_merge['hand'] = hand     # 持仓
                    df_merge['hand'].iloc[0] = 0     # 持仓

                df_merge['pos'] = df_merge['pos']*df_merge['hand']

                df_merge['signal'] = np.where(df_merge['pos']!=df_merge['pos'].shift(-1), df_merge['y_pred'], 0)
                df_merge['signal'].iloc[0] = df_merge['y_pred'].iloc[0]     # 信号

                df_merge['pre_close'] = df_merge['close'].shift(1)
                df_merge['pre_signal'] = df_merge['signal'].shift(1)
                df_merge['pre_signal'].iloc[0], df_merge['pre_close'].iloc[0] = 0, 0

                df_merge['pre_datetime'] = df_merge['datetime'].shift(1)
                df_merge['pre_datetime'].iloc[0] = df_merge['datetime'].iloc[0]

                df_trade = pd.concat([pd.DataFrame(i) for i in df_merge.apply(get_open_price, axis=1)])  # 交易信息
                df_merge['trade_time'], df_merge['trade_price'] = df_trade['trade_time'].values, df_trade['trade_price'].values

                if pricetick_type:
                    df_merge['cost'] = (df_merge['signal'].abs()*(rate+pricetick_rate*real_pricetick)*df_merge['close']*size*df_merge['hand']*2).shift(1)   # 手续费
                else:
                    df_merge['cost'] = (df_merge['signal'].abs()*(rate*df_merge['close']+pricetick_rate*real_pricetick)*size*df_merge['hand']*2).shift(1)   # 手续费
                df_merge['cost'].iloc[0] = 0

                df_merge['profit'] = df_merge['pos'].shift(1)*(df_merge['trade_price'] - df_merge['pre_close']) + \
                                    (df_merge['close'] - df_merge['trade_price'])*df_merge['pos']      # 利润
                df_merge['profit'] = df_merge['profit']*size
                df_merge['profit'].iloc[0] = 0
                df_merge['profit_cost'] = df_merge['profit'] - df_merge['cost']     # 扣除手续费后的利润
            
            except:
                continue

            res_li.append(df_merge.copy())  # 保存每个合约的表

        df_res = pd.concat(res_li, ignore_index=True)
        df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
        df_res = df_res.iloc[:-1]

        df_res['balance'] = df_res['profit_cost'].cumsum() + balance    # 资金
        df_res['pnl_cost'] = df_res['balance'] / balance  # pnl曲线

        return df_res

    def backtesting0(self, symbol, df_contract_li, y_pred):
        '''
        跑回测, 添加了一分钟
        y_pred: datetime y_pred
        '''
        balance = 1_000_000
        # startdate = self.startdate if self.startdate > startdate else startdate
        # enddate = self.enddate if self.enddate < enddate else enddate
        rate, pricetick, size, hand = self.get_backtest_params(symbol)
        hand = int(balance * hand * 0.15 / 300_000)
        res_li = []
        coun = 0
        
        def get_open_price(x):
            '''计算开仓价格'''
            pre_close, pre_signal = x['pre_close'], x['pre_signal']
            if pre_signal == 0:
                return {'trade_time': [None], 'trade_price': [0]}
            price_li = [pre_close, x['close0'], x['close1'], x['close2'], x['close3']]
            trade_price, trade_time = 0, None
            
            dt = x['datetime'].time()
            start_n = 1 if dt == time(21, 0) or dt == time(13, 0) or dt == time(23, 0) or dt == time(9, 0) else 0

            for i in range(start_n, 4):
                if trade_price:
                    break
                else:
                    if pre_signal == -1 and price_li[i] <= x[f'high{i}']: 
                        trade_price, trade_time = max(x[f'open{i}']-pricetick, price_li[i]), x[f'datetime{i}']
                    elif pre_signal == 1 and price_li[i] >= x[f'low{i}']:
                        trade_price, trade_time = min(x[f'open{i}']+pricetick, price_li[i]), x[f'datetime{i}'] 
            
            if trade_price == 0:
                # print('trade_price is 0...', x)
                # input()
                trade_price, trade_time = price_li[-1], x['datetime3']
            
            return {'trade_time': [trade_time], 'trade_price': [trade_price]}

        for df_60 in df_contract_li:
            try:
                df_merge = pd.merge(df_60, y_pred, left_on='datetime', right_on='datetime', how='inner')
                # df_merge.fillna(method='ffill', inplace=True)
                # df_merge.dropna(inplace=True)
                # df_merge = df_merge[df_merge['datetime']<=df_60['datetime'].iloc[-1]]
                # df_1.to_csv('df_1.csv')
                # df_60.to_csv('df_60.csv')
                # print('dddddddddd')
                # input()
                df_merge['y_pred'] = df_merge['y_pred'] - 1 
                # df_mini = df_merge['datetime'].apply(partial(self.set_mini_data, df=df_1))
                # df_merge = pd.concat([df_merge, df_mini], axis=1)       # 获取1分钟k线，用于交易

                df_merge['pos'] = df_merge['y_pred'].replace(to_replace=0, value=None).shift(1)*hand
                df_merge['pos'].iloc[0] = 0     # 持仓

                df_merge['signal'] = np.where(df_merge['pos']!=df_merge['pos'].shift(-1), df_merge['y_pred'], 0)
                df_merge['signal'].iloc[0] = df_merge['y_pred'].iloc[0]     # 信号

                df_merge['pre_close'] = df_merge['close'].shift(1)
                df_merge['pre_signal'] = df_merge['signal'].shift(1)
                df_merge['pre_signal'].iloc[0], df_merge['pre_close'].iloc[0] = 0, 0

                df_trade = pd.concat([pd.DataFrame(i) for i in df_merge.apply(get_open_price, axis=1)])  # 交易信息
                df_merge['trade_time'], df_merge['trade_price'] = df_trade['trade_time'].values, df_trade['trade_price'].values

                df_merge['cost'] = (df_merge['signal'].abs()*(rate*df_merge['close']+0.5*pricetick)*size*hand*2).shift(1)   # 手续费
                df_merge['cost'].iloc[0] = 0

                # df_merge['gap'] = df_merge['open'] - df_merge['close'].shift(1)     # 跳空

                df_merge['profit'] = df_merge['pos'].shift(1)*(df_merge['trade_price'] - df_merge['pre_close']) + \
                                    (df_merge['close'] - df_merge['trade_price'])*df_merge['pos']      # 利润
                df_merge['profit'] = df_merge['profit']*size
                df_merge['profit'].iloc[0] = 0

                # df_merge['pricetick'] = np.where((df_merge['signal'].shift(1)!=0)&(df_merge['gap']*df_merge['signal'].shift(1)<0), 2*pricetick*size*hand, 0)   # 滑点
                # df_merge['profit'] = df_merge['profit'] - df_merge['pricetick']
                # df_merge.drop(['datetime0', 'open0', 'high0', 'low0', 'close0',
                #                'datetime1', 'open1', 'high1', 'low1', 'close1',
                #                'datetime2', 'open2', 'high2', 'low2', 'close2',
                #                'datetime3', 'open3', 'high3', 'low3', 'close3'], axis=1, inplace=True)
                df_merge['profit_cost'] = df_merge['profit'] - df_merge['cost']     # 扣除手续费后的利润
            
            except:
                continue
            # df_merge['coun'] = coun
            # coun += 1

            # df_merge = df_merge.iloc[:-1]
            res_li.append(df_merge.copy())  # 保存每个合约的表

            # df_merge.to_csv('df_merge_i.csv')
            # print(df_merge['datetime'].iloc[0], df_merge['datetime'].iloc[-1])
            # input()
        
        df_res = pd.concat(res_li, ignore_index=True)
        df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
        df_res = df_res.iloc[:-1]

        df_res['balance'] = df_res['profit_cost'].cumsum() + balance    # 资金
        df_res['pnl_cost'] = df_res['balance'] / balance  # pnl曲线

        return df_res

    def all_contract_backtesting(self, symbol, startdate=None, enddate=None, y_pred=None, target_type='drawdown', is_rq=1, is_hand=1, open_method=1):
        '''单品种全合约回测'''
        if startdate==None:
            startdate, enddate = self.startdate, self.enddate
        delay = 15  # 2
        if is_rq:
            load_pa = f'{pa_prefix}/datas/hour_min_merge/'  
        else:
            load_pa = f'{pa_prefix}/datas_sc/hour_min_merge/'
            self.mainconinfo.set_df_maincon()
        df_contract_li = self.mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=delay, load_pa=load_pa, is_concat=0, interval=60)
        # df_contract60_li = self.mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=delay, load_pa=None, is_concat=0, interval=60)
        # df_contract1_li = self.mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=delay, load_pa=None, is_concat=0, interval=1)

        # try:
        y_pred = y_pred[['datetime', 'pred_sig']] if 'pred_sig' in y_pred.columns.to_list() else y_pred[['datetime', 'y_pred']]
        y_pred['datetime'] = pd.to_datetime(y_pred['datetime'])
        y_pred.columns = ['datetime', 'y_pred']
        df_res = self.backtesting(symbol, df_contract_li, y_pred, is_hand=is_hand, open_method=open_method)
        df_res.reset_index(drop=True, inplace=True)
        # df_res.to_csv('df_res.csv')
        # target = self.caculate_sharp_ratio(df_res)
        target = self.get_target(target_type, df_res)
        
        # except:
        #     # print('出错了！！')
        #     target = -3

        return df_res, target 

    def get_target_score(self, df_res, target_type=''):
        '''获取目标分数'''
        if target_type == 'drawdown':
            target = -self.caculate_drawdown(df_res)
            if target == 0 and df_res['pnl_cost'].iloc[-1] == 0:
                print('target为0', target)
                target = -3
        elif target_type == 'max_ddpercent_duration':
            target = self.caculate_max_ddpercent_duration(df_res)
        elif target_type == 'max_ddpercent':
            target = self.caculate_max_ddpercent(df_res)
        elif target_type == 'total_profit':
            target = self.caculate_total_profit(df_res)
        elif target_type == 'zigzag_trend':
            target = self.caculate_zigzag_return(df_res, 'trend')
        elif target_type == 'zigzag_shock':
            target = self.caculate_zigzag_return(df_res, 'shock')
        else:
            target = -10
        
        # if len(df_res[df_res['signal']!=0]) < 20: 
        #     target = -3
        return target

    def symbol_class_backtesting(self, futures_name, startdate=None, enddate=None, y_pred=None, target_type='drawdown'):
        futures_li = SymbolsInfo().get_futures_li(futures_name)
        df_res_all = pd.DataFrame()   
        df_res_li = self.dp.seperate_df_class(y_pred)     # 把y_pred按品种划分
        y_pred['change'] = np.where(y_pred['datetime'] < y_pred['datetime'].shift(1), 1, 0)
        y_li = [0] + y_pred[y_pred['change']==1].index.to_list()
        del y_pred['change']
        print('y_li:', y_li)
        print(len(y_li), len(futures_li))
        input()

        for i in range(len(futures_li)):
            symbol, y_pred_i = futures_li[i], df_res_li[i]
            y_pred_i.reset_index(inplace=True)
            df_res, _ = self.all_contract_backtesting(symbol, startdate=startdate, enddate=enddate, y_pred=y_pred_i, target_type='')
            df_res = df_res.rename(columns={'pnl_cost': f'pnl_cost_{symbol}'})[['datetime', f'pnl_cost_{symbol}']]
            df_res_all = pd.merge(df_res_all, df_res, how='outer', left_on='datetime', right_on='datetime') if len(df_res_all) else df_res
        
        # 合并所有品种的pnl曲线
        df_res_all.sort_values('datetime', ascending=True, inplace=True)    # 排序
        df_res_all.fillna(method='ffill', inplace=True)     # 填充空值
        df_res_all.fillna(1, inplace=True)
        df_res_all['pnl_cost'] = np.sum(df_res_all.iloc[:, 1:], axis=1) / len(futures_li)
        target = self.get_target_score(df_res_all, target_type)
        return df_res_all, target

    def select_bactesting(self, symbol, startdate=None, enddate=None, y_pred=None, target_type='drawdown'):
        '''判断回测是按板块回测还是按每个品种回测'''
        if startdate==None:
            startdate, enddate = self.startdate, self.enddate

        if symbol not in self.syinfo.symbol_li:
            df_res, target = self.symbol_class_backtesting(symbol, startdate=startdate, enddate=enddate, y_pred=y_pred, target_type=target_type)
        else:
            df_res, target = self.all_contract_backtesting(symbol, startdate=startdate, enddate=enddate, y_pred=y_pred, target_type=target_type)
        return df_res, target

    def progress_backtest_datas(self, save_pa=f'{pa_prefix}/datas/hour_min_merge/'):
        '''60分钟和后面1分钟拼接, 用于回测'''
        delay = 20
        symbol_li = self.syinfo.symbol_li
        makedir(save_pa)
        startdate, enddate = datetime(2016, 1, 1), datetime(2022, 12, 14)
        if 'datas_sc' in save_pa.split('/'):
            self.mainconinfo.set_df_maincon()
            load_pa60 = f'{pa_prefix}/datas_sc/data_60m/'
            load_pa1 = f'{pa_prefix}/datas_sc/data_1m/'
        else:
            load_pa60 = None
            load_pa1 = None

        for symbol in symbol_li:
            try:
                df_res = self.mainconinfo.get_symbol_df_maincon(symbol, startdate, enddate, delay, delay_end=0)

                df_contract60_li = self.mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=delay, load_pa=load_pa60, is_concat=0, interval=60)
                df_contract1_li = self.mainconinfo.get_main_contact_k_line(symbol, startdate, enddate, delay=delay, load_pa=load_pa1, is_concat=0, interval=1)
                for i in range(len(df_res)): 
                    df_1, df_60 = df_contract1_li[i], df_contract60_li[i]
                    contract = df_res['contract'].iloc[i]
                    df_mini = df_60['datetime'].apply(partial(self.set_mini_data, df=df_1))
                    df_merge = pd.concat([df_60, df_mini], axis=1)       # 获取1分钟k线，用于交易
                    save_pa_sy = makedir(f'{save_pa}{symbol}')
                    df_merge.to_csv(f'{save_pa_sy}/{contract}.csv', index=False)
                    print(contract, 'done.')
            except:
                print('没有该品种', symbol)

    def signal_analyze(self, symbol, pred_pa, traindate=None, valdate=None, testdate=None, enddate=None, save_df=1, params='tvt'):
        '''信号结果分析'''
        capital = 1_000_000

        # y_pred = './filter_results/ru/res2/y_pred_[5, 0.5, 1, 1]_ru_60m_0.6_50_1_return_rate_60m'
        # y_pred = './filter_results/v/res2/y_pred_[5, 0.5, 1, 0]_v_60m_0.6_50_1_return_rate_60m'
        if traindate is None:
            # traindate, valdate, testdate = datetime(2016, 1, 1), datetime(2019, 5, 1), datetime(2020, 5, 1)
            # enddate = datetime(2020, 10, 30)
            traindate, valdate, testdate = self.startdate, datetime(2019, 5, 1), datetime(2020, 5, 1)
            enddate = self.enddate
        
        save_pa=f'{pa_prefix}/datas/backtest_res/{symbol}/'
        makedir(save_pa)
        suffix = pred_pa.split('/')[-1][:-4]
        save_pa = f'{save_pa}{suffix}'

        # rate, pricetick, size, hand = self.get_backtest_params(symbol)  # 获取symbol相关属性

        if params == 'all':
            date_dic = {'all': [traindate, enddate]}
        elif params == 'tvt':
            date_dic = {'train': [traindate, valdate], 'val': [valdate, testdate], 'test': [testdate, enddate]}
        elif params == 'val':
            date_dic = {'val': [valdate, testdate]}

        data_li = []
        for i in date_dic:
            df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, date_dic[i][0], date_dic[i][1], delay=2)
            res = []
            # trade_res = []
            for j in range(df_contractinfo.shape[0]):
                q = df_contractinfo.iloc[j].to_list()
                print(q)
                # input()
                engine, _ = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), pred_pa, plot=0, need_engine=1)
                if engine is 0:
                    continue
                df_i = pd.DataFrame(engine.strategy.res_dic)
                res.append(df_i)
            #     for k in engine.trades.values():      # 获取交易记录
            #         trade_res.append([k.datetime, k.direction, k.offset, k.price])
            # trade_res = pd.DataFrame(trade_res)
            # trade_res.columns = ['datetime', 'direction', 'offset', 'price']
            # trade_res.to_csv(f'{save_pa}_{i}_trade.csv', index=False)
            df_res = pd.concat(res)
            df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
            df_res['pnl'] = df_res['profit'].cumsum() / capital + 1
            df_res['pnl_cost'] = (df_res['profit']-df_res['cost']).cumsum() / capital + 1
            data_li.append(df_res[['datetime', 'pnl_cost']].copy())
            # print(df_res[['datetime', 'pnl_cost']].copy())
            # print('len df_res:', len(df_res))

            if save_df:
                df_res.to_csv(f'{save_pa}_{i}_analyze.csv', index=False)
                # df_res.to_csv(f'analyze.csv', index=False)
        
        train_pa = f'{save_pa}_train_analyze'
        ms = ModelStatistics()  # 统计回测结果
        # ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_statistics', symbol=symbol)
        try:
            ms.caculate_statistics_all(train_pa=train_pa, save_pa=f'{save_pa}_statistics', symbol=symbol)
        except:
            print(train_pa, 'statistics got wrong.')
        ms.caculate_statistics_trend_shock_train_val_test(symbol, train_pa, f'{save_pa}_trend_shock_statistic')
        
        train_val_test_pnl_plot(data_li, save_pa)   # 将三个pnl曲线画在一张图并保存
        
        return df_res

    def all_symbols_backtesting(self, symbol_li, startdate=None, enddate=None, y_pred_li=None, params={}, delay=0, target_type='drawdown', save_pa=''):
        '''全品种回测'''
        str_date = f'{enddate}'.split(' ')[0]
        df_res_li = []
        df_res_all = pd.DataFrame()     
        is_rq = 0 if 'datas_sc' in save_pa.split('/') else 1
        # 对所有品种跑回测
        for i in range(len(symbol_li)):
            symbol = symbol_li[i]
            y_pred = y_pred_li if y_pred_li is None else y_pred_li[i]
            df_res, _ = self.all_contract_backtesting(symbol, startdate=startdate, enddate=enddate, y_pred=y_pred, 
                                target_type=target_type, is_rq=is_rq, is_hand=1, open_method=0)
            # df_res.to_csv(f'{save_pa}df_val_{symbol}.csv')
            df_res.to_csv(f'{save_pa}df_{str_date}_{symbol}.csv')

            # exit()
            df_res.rename(columns={'pnl_cost': f'pnl_cost_{symbol}'}, inplace=True)
            df_res_li.append(df_res)

        # 对所有品种的pnl进行merge
        for j in range(len(df_res_li)):
            df_i = df_res_li[j][['datetime', f'pnl_cost_{symbol_li[j]}']]
            if len(df_res_all) == 0:
                df_res_all = df_i
            else:
                df_res_all = pd.merge(df_res_all, df_i, how='outer', left_on='datetime', right_on='datetime')

        # 合并所有品种的pnl曲线
        df_res_all.sort_values('datetime', ascending=True, inplace=True)    # 排序
        df_res_all.fillna(method='ffill', inplace=True)     # 填充空值
        df_res_all.fillna(1, inplace=True)
        df_res_all['pnl_cost_total'] = np.sum(df_res_all.iloc[:, 1:], axis=1) / len(symbol_li)

        df_res_all.to_csv('df_res_all.csv')
        # exit()
        # 获取目标值
        df_res = df_res_all[['datetime', 'pnl_cost_total']].copy()
        df_res.rename(columns={'pnl_cost_total': 'pnl_cost'}, inplace=True)
        target = self.get_target(target_type, df_res)

        return target, df_res_all 

def MyBackTester_debug():
    '''自己的回测和1.0比'''
    pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/'
    pa_str = './datas/ml_result/model_1.0/test/'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/[7, 0.6, 1, 1]_AP_60m_1.5_sample_3_1_return_rate_60m/y_pred_[7, 0.6, 1, 1]_AP_60m_1.5_sample_3_1_return_rate_60m_train_analyze.csv'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index_adj1/[10, 0.5, 1, 1]_C_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 0.5, 1, 1]_C_60m_0.7_sample_10_1_return_rate_60m_test_analyze.csv'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m_train_analyze.csv'
    pa_my = './df_res.csv'
    mbt = MyBackTester()
    bt = BackTester()
    # symbol = 'JD'
    pa_li = os.listdir(pa_str)
    # for i, j in enumerate(pa_li):
    #     print(i, j)
    # input()
    for pa_i in pa_li[:]:
        sy_pa_li = os.listdir(f'{pa_str}{pa_i}')
        load_pa = filter_str(f'train_analyze.csv', sy_pa_li)
        symbol = pa_i.split('_')[1]
        print('begin:', symbol)
        y_pred = pd.read_csv(f'{pa_str}{pa_i}/{load_pa}')
        df_res = y_pred.copy()
        df_res['datetime'] = pd.to_datetime(df_res['datetime'])
        target = bt.get_target(target_type='drawdown', df_res=df_res)
        y_pred['pred_sig'] = y_pred['pred_sig'] + 1
        y_pred.rename(columns={'pred_sig': 'y_pred'}, inplace=True)
        st = datetime.now()
        # print(df_res.tail(20))
        time0 = datetime.now()-st
        # input()
        df_res.to_csv('df_res.csv')
        df_res = pd.read_csv('df_res.csv')
        df_res['pred_sig'] = df_res['pred_sig'] + 1
        df_res.rename(columns={'pred_sig': 'y_pred'}, inplace=True)
        st = datetime.now()
        df_res1, target1 = mbt.all_contract_backtesting(symbol, startdate=df_res['datetime'].iloc[0], enddate=df_res['datetime'].iloc[-1], y_pred=df_res, target_type='drawdown', 
                                )
        time1 = datetime.now()-st
        # print(df_res1.tail(20))
        # input()
        
        # target = -bt.caculate_drawdown(df_res)
        res0, res1 = df_res['pnl_cost'].iloc[-1], df_res1['pnl_cost'].iloc[-1]
        print(symbol, res0, res1, res1-res0, target, target1, target1-target, time0, time1)  # 

    # df_res.to_csv('df_res.csv')
    df_res1.to_csv('df_res1.csv')
    # df_res = pd.read_csv('df_res.csv')
    df_res1 = pd.read_csv('df_res1.csv')
    df_res.set_index('datetime', inplace=True)
    df_res1.set_index('datetime', inplace=True)
    df_merge = pd.merge(df_res, df_res1, left_index=True, right_index=True, how='outer')
    df_merge[['pnl_cost_x', 'pnl_cost_y']].plot()
    plt.show()
    df_merge.to_csv('df_merge.csv')

    print('done.')

def MyBackTester_debug1():
    '''自己的回测和vnpy比'''
    pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/'
    pa_str = './datas/ml_result/model_1.0/test/'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/[7, 0.6, 1, 1]_AP_60m_1.5_sample_3_1_return_rate_60m/y_pred_[7, 0.6, 1, 1]_AP_60m_1.5_sample_3_1_return_rate_60m_train_analyze.csv'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index_adj1/[10, 0.5, 1, 1]_C_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 0.5, 1, 1]_C_60m_0.7_sample_10_1_return_rate_60m_test_analyze.csv'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m_train_analyze.csv'
    pa_my = './df_res.csv'
    mbt = MyBackTester()
    bt = BackTester()
    # symbol = 'JD'
    pa_li = os.listdir(pa_str)
    # for i, j in enumerate(pa_li):
    #     print(i, j)
    # input()
    for pa_i in pa_li[:]:
        sy_pa_li = os.listdir(f'{pa_str}{pa_i}')
        load_pa = filter_str(f'train_analyze.csv', sy_pa_li)
        symbol = pa_i.split('_')[1]
        print('begin:', symbol)
        y_pred = pd.read_csv(f'{pa_str}{pa_i}/{load_pa}')
        # df_res = y_pred.copy()
        y_pred['pred_sig'] = y_pred['pred_sig'] + 1
        y_pred.rename(columns={'pred_sig': 'y_pred'}, inplace=True)
        st = datetime.now()
        df_res, target = bt.all_contract_backtesting(symbol, startdate=str_to_datetime(y_pred['datetime'].iloc[0]), 
                                                enddate=str_to_datetime(y_pred['datetime'].iloc[-1]), y_pred=y_pred, target_type='drawdown')
        # print(df_res.tail(20))
        time0 = datetime.now()-st
        # input()
        df_res.to_csv('df_res.csv')
        df_res = pd.read_csv('df_res.csv')
        df_res['pred_sig'] = df_res['pred_sig'] + 1
        df_res.rename(columns={'pred_sig': 'y_pred'}, inplace=True)
        st = datetime.now()
        df_res1, target1 = mbt.all_contract_backtesting(symbol, startdate=df_res['datetime'].iloc[0], enddate=df_res['datetime'].iloc[-1], y_pred=df_res, target_type='drawdown',
                                    is_hand=1, open_method=0)
        time1 = datetime.now()-st
        # print(df_res1.tail(20))
        # input()
        
        # target = -bt.caculate_drawdown(df_res)
        res0, res1 = df_res['pnl_cost'].iloc[-1], df_res1['pnl_cost'].iloc[-1]
        print(symbol, res0, res1, res1-res0, target, target1, target1-target, time0, time1)  # 

    # df_res.to_csv('df_res.csv')
    df_res1.to_csv('df_res1.csv')
    # df_res = pd.read_csv('df_res.csv')
    df_res1 = pd.read_csv('df_res1.csv')
    df_res.set_index('datetime', inplace=True)
    df_res1.set_index('datetime', inplace=True)
    df_merge = pd.merge(df_res, df_res1, left_index=True, right_index=True, how='outer')
    df_merge[['pnl_cost_x', 'pnl_cost_y']].plot()
    plt.show()
    df_merge.to_csv('df_merge.csv')

    print('done.')

def MyBackTester_debug2():
    '''回测和1.0比'''
    pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/'
    pa_str = './datas/ml_result/model_1.0/test/'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/[7, 0.6, 1, 1]_AP_60m_1.5_sample_3_1_return_rate_60m/y_pred_[7, 0.6, 1, 1]_AP_60m_1.5_sample_3_1_return_rate_60m_train_analyze.csv'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index_adj1/[10, 0.5, 1, 1]_C_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 0.5, 1, 1]_C_60m_0.7_sample_10_1_return_rate_60m_test_analyze.csv'
    # pa_str = './datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m_train_analyze.csv'
    pa_my = './df_res.csv'
    mbt = MyBackTester()
    bt = BackTester()
    # symbol = 'JD'
    pa_li = os.listdir(pa_str)
    # for i, j in enumerate(pa_li):
    #     print(i, j)
    # input()
    for pa_i in pa_li[:]:
        sy_pa_li = os.listdir(f'{pa_str}{pa_i}')
        load_pa = filter_str(f'train_analyze.csv', sy_pa_li)
        symbol = pa_i.split('_')[1]
        print('begin:', symbol)
        y_pred = pd.read_csv(f'{pa_str}{pa_i}/{load_pa}')
        df_res1 = y_pred.copy()
        df_res1['datetime'] = pd.to_datetime(df_res1['datetime'])
        target1 = 0
        y_pred['pred_sig'] = y_pred['pred_sig'] + 1
        y_pred.rename(columns={'pred_sig': 'y_pred'}, inplace=True)
        st = datetime.now()
        df_res, target = bt.all_contract_backtesting(symbol, startdate=str_to_datetime(y_pred['datetime'].iloc[0]), 
                                                enddate=str_to_datetime(y_pred['datetime'].iloc[-1]), y_pred=y_pred, target_type='drawdown')
        # print(df_res.tail(20))
        time0 = datetime.now()-st
        # input()
        df_res.to_csv('df_res.csv')
        df_res = pd.read_csv('df_res.csv')
        df_res['pred_sig'] = df_res['pred_sig'] + 1
        df_res.rename(columns={'pred_sig': 'y_pred'}, inplace=True)
        st = datetime.now()
        # df_res1, target1 = mbt.all_contract_backtesting(symbol, startdate=df_res['datetime'].iloc[0], enddate=df_res['datetime'].iloc[-1], y_pred=df_res, target_type='drawdown')
        time1 = datetime.now()-st
        # print(df_res1.tail(20))
        # input()
        
        # target = -bt.caculate_drawdown(df_res)
        res0, res1 = df_res['pnl_cost'].iloc[-1], df_res1['pnl_cost'].iloc[-1]
        print(symbol, res0, res1, res1-res0, target, target1, target1-target, time0, time1)  # 

    # df_res.to_csv('df_res.csv')
    df_res1.to_csv('df_res1.csv')
    # df_res = pd.read_csv('df_res.csv')
    df_res1 = pd.read_csv('df_res1.csv')
    df_res.set_index('datetime', inplace=True)
    df_res1.set_index('datetime', inplace=True)
    df_merge = pd.merge(df_res, df_res1, left_index=True, right_index=True, how='outer')
    df_merge[['pnl_cost_x', 'pnl_cost_y']].plot()
    plt.show()
    df_merge.to_csv('df_merge.csv')

    print('done.')

def run_signal_analyze1():
    bt = BackTester()
    pa = f'{pa_prefix}/datas/ml_result/model_2.0/max_drawdown_skew_skew_10_7/params/'
    pa_li = os.listdir(pa)  
    for sy in ['M']:
        load_pa_li = os.listdir(f'{pa}{sy}/')
        for lp in load_pa_li:
            print(sy, lp)
            load_f = f'{pa}{sy}/{lp}/'
            pred_pa_li = filter_str('y_pred', os.listdir(load_f), is_list=1)
            pred_pa = filter_str('_60m.csv', pred_pa_li, is_list=0)
            pred_pa = f'{load_f}{pred_pa}'
            bt.signal_analyze1(sy, pred_pa, lp, params='tvt')



def run():
    s = BackTester()
    sy = 'm'
    li = [f'y_pred_{i}_m_60m_1.5_1.533_sample_20_1_return_rate_60m' for i in ['[10, 1, 1, 1]', '[5, 1, 1, 1]', '[10, 0.5, 1, 1]', '[5, 0.5, 1, 1]', '[7, 0.6, 1, 1]']]
    pa0 = './datas/ml_result/symbol_result_10_index/params/'
    pa1 = f'{pa0}[7, 0.6, 1, 1]_PP_60m_1.4_sample_10_1_return_rate_60m/y_pred_[7, 0.6, 1, 1]_PP_60m_1.4_sample_10_1_return_rate_60m.csv'
    pa3 = f'{pa0}[5, 0.5, 1, 1]_JD_60m_1.3_sample_10_1_return_rate_60m/y_pred_[5, 0.5, 1, 1]_JD_60m_1.3_sample_10_1_return_rate_60m.csv'
    pa2 = f'{pa0}[10, 1, 1, 1]_RM_60m_1.4_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_RM_60m_1.4_sample_10_1_return_rate_60m.csv'
    
    # pa0 = './datas/ml_result/symbol_result_10_index_adj/params/'
    # pa1 = f'{pa0}[5, 0.5, 1, 1]_PP_60m_1.3_sample_6_1_return_rate_60m/y_pred_[5, 0.5, 1, 1]_PP_60m_1.3_sample_6_1_return_rate_60m.csv'
    # pa2 = f'{pa0}[7, 0.6, 1, 1]_RM_60m_1.1_sample_9_1_return_rate_60m/y_pred_[7, 0.6, 1, 1]_RM_60m_1.1_sample_9_1_return_rate_60m.csv'
    # pa3 = f'{pa0}[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_JD_60m_0.7_sample_10_1_return_rate_60m.csv'
    pa_li = [pa1, pa2, pa3]
    symbol_li = ['PP', 'RM', 'JD']
    for symbol, pred_pa in zip(symbol_li, pa_li):
        s.signal_analyze(symbol, pred_pa, traindate=None, valdate=None, testdate=None, enddate=None, save_df=1, params='tvt')
    
def run1():
    mbt = MyBackTester()
    mbt.progress_backtest_datas(f'{pa_prefix}/datas/hour_min_merge/')

if __name__ == "__main__":
    # MyBackTester_debug1()
    # run_signal_analyze1()
    # MyBackTester_debug()
    run1()
    # for i in li:
    #     print(i)
    #     s.signal_analyze(sy, f'{pa_prefix}/filter_results/{sy}/res8/{i}')
    # s.signal_analyze('v', './datas/predict/v/y_pred_[5, 0.5, 1, 0]_v_60m_1.2_20_1_return_rate_60m')
    # s.backtesting_all()
    # s.backtesting('RB')
    # y_pred_[5, 0.5, 1, 0]_rb_60m_0.6_50_1_return_rate_60m
    # y_pred_[5, 0.5, 1, 1]_ru_60m_0.6_50_1_return_rate_60m
    # y_pred_[5, 0.5, 1, 0]_fg_60m_0.6_50_1_return_rate_60m
    # y_pred_[5, 0.7, 1, 1]_j_60m_0.6_50_1_return_rate_60m
    '''--------------------------------------------------'''
    # y_pred_[5, 0.5, 1, 1]_ru_60m_0.6_50_1_return_rate_60m
    # y_pred_[5, 0.5, 1, 0]_rb_60m_0.6_50_1_return_rate_60m
    # y_pred_[5, 0.5, 1, 0]_v_60m_0.6_50_1_return_rate_60m
    # D:\策略开发\futures_ml\datas\predict\v
    # s.all_contract_backtesting_1('v', startdate=datetime(2016, 5, 1), enddate=datetime(2020, 10, 30), y_pred='./datas/predict/v/y_pred_[5, 0.5, 1, 0]_v_60m_1.2_20_1_return_rate_60m')



  