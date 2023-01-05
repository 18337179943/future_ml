#%%
import sys, os
from m_base import *
sys_name = 'windows'
pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from datetime import datetime
from datas_process.m_futures_factors import SymbolsInfo, MainconInfo
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
import matplotlib.pyplot as plt
from m_base import *
from backtesting import BackTester
from simulation.simulation_strategy import SimulationStrategy   # 只是n倍atr
from simulation.simulation_strategy_index import SimulationIndexStrategy   # 只是n倍atr
from simulation.base_strategy import BaseStrategy  # 均线加n倍atr
# from backtesting.ml_strategy import MLStrategy
# from backtesting.ml_strategy_adj import MLStrategy
from backtesting.ml_strategy_adj import MLStrategy
# from atr_rsi_strategy import AtrRsiStrategy
from m_base import Logger, get_sy, timestamp_to_datetime, train_val_test_pnl_plot
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
sys.path.append('..')
from m_base import str_to_datetime
# sys.stdout = Logger('{pa_prefix}/datas/backtest_res/log.txt')


class SimulationBackTester(BackTester):
    '''
    1、百分比止盈止损
    2、百分比回撤止盈
    3、k倍atr止盈止损失
    4、固定时间缩小止盈
    5、逐笔加减仓
    '''
    def __init__(self, startdate=datetime(2016, 5, 20), enddate=datetime(2022, 11, 20), strategy_class=BaseStrategy): # datetime(2020, 10, 30)
        super().__init__(startdate, enddate)
        self.res_pa = f'{pa_prefix}/datas/simulation_res/'
        self.pred_pa = f'{pa_prefix}/datas/predict/'
        self.strategy_class = strategy_class

    def backtesting(self, contract, startdate, enddate, y_pred, plot=True, need_engine=0, params={}):
        '''跑回测'''
        symbol = get_sy(contract)
        rate, pricetick, size, hand = self.get_backtest_params(symbol)
        # rate = 0.0004
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
            capital=self.capital,
        )
        params0 = {'hand': hand, 'symbol_name': symbol, 'contract': contract, 'y_pred': y_pred,    # 'y_pred_[10, 1, 1, 0]_ru_5m_0.6_50_1_return_rate_60m'
                    'sig_meth': self.sig_meth, 'step_n': 1, 'win_n': 1, 'rate': rate, 'size': size, 'pricetick': pricetick}
        params.update(params0)
        engine.add_strategy(self.strategy_class, params)
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

    def all_contract_backtesting(self, symbol, startdate=None, enddate=None, y_pred=None, params={}, delay=0, target_type='drawdown'):
        '''单品种全合约回测'''
        if startdate==None:
            startdate, enddate = self.startdate, self.enddate
        df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, startdate, enddate, delay=delay)
        annual_return_li, res_li = [], []
        for i in range(df_contractinfo.shape[0]):
            q = df_contractinfo.iloc[i].to_list()
            # print('begin: ', q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]))
            if i == 0:
                params.update({'init_balance': 1_000_000})  # 初始化资金
                params.update({'profit_rate_li': []})  # 交易结果
            else:
                if not isinstance(engine, int):
                    params.update({'profit_rate_li': engine.strategy.profit_rate_li})

                try:
                    params.update({'init_balance': df_i['balance'].iloc[-1]})
                except:
                   params.update({'init_balance': 1_000_000}) 
            # print(q)
            # print(startdate,enddate)
            # input()
            engine, res = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), y_pred, plot=False, need_engine=1, params=params)
            
            if engine is 0:
                print(q)
                continue
            df_i = pd.DataFrame(engine.strategy.res_dic)
            annual_return_li.append(df_i)
            res_li.append(res['total_return'])
        # try:
        # target = np.sum(res_li)
        df_res = pd.concat(annual_return_li)
        df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
        df_res['profit_cost'] = df_res['profit'] - df_res['cost']
        df_res['pnl_cost'] = df_res['profit_cost'].cumsum() / self.capital + 1
        df_res.reset_index(drop=True, inplace=True)
        # target = self.caculate_sharp_ratio(df_res)
        target = self.get_target(target_type, df_res)
            
        if target == 0 and df_res['pnl_cost'].iloc[-1] == 0:
            print('target为0', target)
            target = -3
            del engine
        # except:
        #     # print('出错了！！')
        #     df_res = pd.DataFrame()
        #     target = -3

        return df_res, target

    def all_symbols_backtesting(self, symbol_li, startdate=None, enddate=None, y_pred_li=None, params={}, delay=0, target_type='drawdown', save_pa=None):
        '''全品种回测'''
        str_date = f'{enddate}'.split(' ')[0]
        df_res_li = []
        df_res_all = pd.DataFrame()     
        # 对所有品种跑回测
        for i in range(len(symbol_li)):
            symbol = symbol_li[i]
            y_pred = y_pred_li if y_pred_li is None else y_pred_li[i]
            df_res, _ = self.all_contract_backtesting(symbol, startdate=startdate, enddate=enddate, y_pred=y_pred, 
                                params=params, delay=delay, target_type=target_type)
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

        # 获取目标值
        df_res = df_res_all[['datetime', 'pnl_cost_total']].copy()
        df_res.rename(columns={'pnl_cost_total': 'pnl_cost'}, inplace=True)
        target = self.get_target(target_type, df_res)

        return target, df_res_all 

    def signal_analyze_total(self, symbol_li, y_pred_li=[], traindate=None, valdate=None, testdate=None, enddate=None, save_df=1, params={}, folder_name='total', sp=''):
        '''信号结果分析'''
        if traindate is None:
            # traindate, valdate, testdate = datetime(2018, 1, 1), datetime(2019, 5, 1), datetime(2020, 10, 31)
            # enddate = datetime(2021, 12, 31)  # datetime(2020, 10, 31)
            traindate, valdate, testdate = datetime(2017, 1, 1), datetime(2019, 5, 1), datetime(2020, 5, 1)
            enddate = datetime(2020, 11, 1)  # datetime(2020, 10, 31)

        self.save_params = makedir(f'{pa_prefix}/simulation/optuna_params/{folder_name}/') if len(sp) == 0 else sp
        save_pa = f'{self.save_params}{folder_name}'

        date_dic = {'train': [traindate, valdate], 'val': [valdate, testdate], 'test': [testdate, enddate]}

        data_li = []
        df_res_all_li = []
        for i in date_dic:
            # i = 'test'
            _, df_res_all = self.all_symbols_backtesting(symbol_li, startdate=date_dic[i][0], enddate=date_dic[i][1], y_pred_li=y_pred_li, params=params, delay=2, target_type='total_profit', save_pa=sp)

            # df_res_all = pd.DataFrame()     
            # # 对所有品种跑回测
            # _, df_res_li = self.all_symbols_backtesting(symbol_li, startdate=date_dic[i][0], enddate=date_dic[i][1], y_pred_li=y_pred_li, params=params, delay=20, target_type='total_profit')
            # # 对所有品种的pnl进行merge
            # for j in range(len(df_res_li)):
            #     df_i = df_res_li[j][['datetime', f'pnl_cost_{symbol_li[j]}']]
            #     if len(df_res_all) == 0:
            #         df_res_all = df_i
            #     else:
            #         df_res_all = pd.merge(df_res_all, df_i, how='outer', left_on='datetime', right_on='datetime')

            # # 合并所有品种的pnl曲线
            # df_res_all.sort_values('datetime', ascending=True, inplace=True)    # 排序
            # df_res_all.fillna(method='ffill', inplace=True)     # 填充空值
            # df_res_all.fillna(1, inplace=True)
            # df_res_all['pnl_cost_total'] = np.sum(df_res_all.iloc[:, 1:], axis=1) / len(symbol_li)

            # 保存数据表
            if save_df:
                df_res_all.to_csv(f'{save_pa}_{i}_analyze.csv', index=False)
            
            data_li.append(df_res_all[['datetime', 'pnl_cost_total']])
            df_res_all_li.append(df_res_all)

        train_val_test_pnl_plot(data_li, save_pa)   # 将三个pnl曲线画在一张图并保存
        return df_res_all_li

    def signal_analyze(self, symbol, pred_pa, traindate=None, valdate=None, testdate=None, enddate=None, save_df=1, params={}):
        '''信号结果分析'''
        # y_pred = f'{pa_prefix}/filter_results/ru/res2/y_pred_[5, 0.5, 1, 1]_ru_60m_0.6_50_1_return_rate_60m'
        # y_pred = f'{pa_prefix}/filter_results/v/res2/y_pred_[5, 0.5, 1, 0]_v_60m_0.6_50_1_return_rate_60m'
        if traindate is None:
            traindate, valdate, testdate = datetime(2016, 5, 1), datetime(2019, 5, 1), datetime(2020, 5, 1)
            enddate = datetime(2020, 10, 31)

        self.save_params = makedir(f'{pa_prefix}/simulation/optuna_params/{symbol}/')
        save_pa=self.save_params
        if isinstance(pred_pa, str):
            suffix = pred_pa.split('/')[-1][:-4]
        else:
            suffix = f'{symbol}_final_test'
        save_pa = f'{save_pa}{suffix}'

        # rate, pricetick, size, hand = self.get_backtest_params(symbol)  # 获取symbol相关属性

        date_dic = {'train': [traindate, valdate], 'val': [valdate, testdate], 'test': [testdate, enddate]}

        data_li = []
        data_price_li = []
        for i in date_dic:
            df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, date_dic[i][0], date_dic[i][1], delay=0)
            res = []
            # trade_res = []
            for j in range(df_contractinfo.shape[0]):
                q = df_contractinfo.iloc[j].to_list()
                print(q)
                engine, _ = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), pred_pa, plot=0, need_engine=1, params=params)
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
            df_res['pnl'] = df_res['profit'].cumsum() / self.capital + 1
            df_res['pnl_cost'] = (df_res['profit']-df_res['cost']).cumsum() / self.capital + 1
            data_li.append(df_res[['datetime', 'pnl_cost']].copy())
            data_price_li.append(df_res[['datetime', 'close']].copy())
            # print(df_res[['datetime', 'pnl_cost']].copy())
            # print('len df_res:', len(df_res))

            if save_df:
                df_res.to_csv(f'{save_pa}_{i}_analyze.csv', index=False)
                # df_res.to_csv(f'analyze.csv', index=False)

        train_val_test_pnl_plot(data_li, save_pa)   # 将三个pnl曲线画在一张图并保存
        train_val_test_pnl_plot(data_price_li, f'{save_pa}_price')   # 将三个价格曲线画在一张图并保存

        return df_res
    
    # def change_pos_month(self, load_pa, symbol_li):
    #     df_merge = pd.DataFrame()
    #     for symbol in symbol_li:
    #         df = pd.read_csv(f'{load_pa}df_test_{symbol}.csv').iloc[:, 1:]
    #         df.set_index('datetime', inplace=True)
    #         [df.rename(columns={col: f'{col}_{symbol}'}, inplace=True) for col in df.columns]
    #         df_merge = pd.merge(df_merge, df, left_index=True, right_index=True, how='outer')
    #     df_merge.fillna(method='ffill', inplace=True)
    #     df_merge['balance_total'] = df_merge[[f'balance_{sy}' for sy in symbol_li]].sum(axis=1)

    #     index_month = str_to_datetime(df_merge.index[0]).month
    #     for i in len(df_merge):
            


if __name__ == "__main__":
    s = BackTester()
    
    sy = 'm'
    li = [f'y_pred_{i}_m_60m_1.5_1.533_sample_20_1_return_rate_60m' for i in ['[10, 1, 1, 1]', '[5, 1, 1, 1]', '[10, 0.5, 1, 1]', '[5, 0.5, 1, 1]', '[7, 0.6, 1, 1]']]
    for i in li:
        print(i)
        s.signal_analyze(sy, f'/filter_results/{sy}/res8/{i}')
    # s.signal_analyze('v', '{pa_prefix}/datas/predict/v/y_pred_[5, 0.5, 1, 0]_v_60m_1.2_20_1_return_rate_60m')
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
    # s.all_contract_backtesting_1('v', startdate=datetime(2016, 5, 1), enddate=datetime(2020, 10, 30), y_pred='{pa_prefix}/datas/predict/v/y_pred_[5, 0.5, 1, 0]_v_60m_1.2_20_1_return_rate_60m')



  