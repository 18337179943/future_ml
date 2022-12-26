from datetime import timedelta, datetime, time
import imp
from string import printable
from tkinter import SEL
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
    Direction
)
import sys
from m_base import *
sys_name = 'windows'
pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import pandas as pd

import numpy as np
from vnpy.trader.constant import Interval
import random
# sys.path.append("..")



class ArrayManager_m(ArrayManager):
    def __init__(self, size: int = 100):
        super().__init__(size)
        """Constructor"""
        self.datetime_array = np.empty(size, dtype=object)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        super().update_bar(bar)
        self.datetime_array[:-1] = self.datetime_array[1:]
        self.datetime_array[-1] = bar.datetime

    @property
    def datetime(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.datetime_array


class MLStrategy(CtaTemplate):
    """"""

    author = "用Python的交易员"

    hand = 1    # 手数
    symbol_name = 'IF'  # 合约名称
    step_n = 12     # 预测步数
    y_pred = 'y_pred'
    pa = 'test_datas_sharp_ratio_30m_5m'
    sig_meth = 0
    win_n = 5
    rate = 0
    size = 10
    pricetick = 0

    sig = 0         # 当前信号
    pre_sig = 0

    parameters = [
        "hand",
        "symbol_name",
        "step_n",
        "y_pred",
        "pa",
        "sig_meth",
        "win_n",
        "rate",
        "size",
        "pricetick"
    ]
    variables = [
        "sig",
        "pre_sig"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        it = Interval.MINUTE if self.win_n != 1 else Interval.HOUR
        self.bg = BarGenerator(self.on_bar)
        self.bgn = BarGenerator(self.on_bar, self.win_n, self.on_n_bar, interval=it)
        self.am = ArrayManager_m(100)
        self.amn = ArrayManager_m(4)
        self.is_macon_change = 0
        self.count = 0
        self.res_dic = {'datetime': [], 'close': [], 'trade_price': [], 'trade_time': [], 'signal': [], 'pos': [], 'profit': [], 'cost': []}
        self.first = 1
        self.pos_1 = 0
        self.count_sig = random.randint(0,1)
        self.beign_trade = 0

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        df = pd.read_csv(f'{pa_prefix}/datas/maincon.csv') 
        self.df_ma = df[df['symbol'] == self.symbol_name].copy()     # 获取品种主力合约数据
        
        # self.df_ma = df[df['is_change']==1]     # 判断合约换月
        # self.df_ma['date'] = self.df_ma['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # self.df_pred = pd.read_csv(f'{pa_prefix}/datas/data_set/{self.symbol_name}/{self.pa}.csv')
        if isinstance(self.y_pred, str):
            # self.df_pred = pd.read_csv(f'{pa_prefix}/datas/predict/{self.symbol_name}/{self.y_pred}.csv')    # 获取预测信号
            self.df_pred = pd.read_csv(f'{self.y_pred}')    # 获取预测信号
            if 'y' in self.df_pred.columns.to_list():
                self.df_pred = self.df_pred.rename(columns={'y': 'y_pred'})
        else:
            self.df_pred = self.y_pred
        # self.df_pred = self.df_pred[self.df_pred['symbol']==self.symbol_name]
        # self.df_pred['datetime'] = self.df_pred['datetime'].apply(pd.Timestamp)

        self.load_bar(1)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def transfrom_sig(self, df):
        if self.sig_meth == 0:
            '''转换信号'''
            sig = df['y_pred'].iloc[0] - 1
            # if sig == 0:
            #     sig = -1
            # elif sig == 1:
            #     sig = 0
            # else:
            #     sig = 1
        elif self.sig_meth == 1:
            '''概率转换'''
            decline, zero, rise = df['decline'].iloc[0], df['zero'].iloc[0], df['rise'].iloc[0]
            max_prob = max(max(decline, zero), rise)
            prob_thread = 0.4
            if max_prob == decline and decline > prob_thread:
                sig = -1
            elif max_prob == rise and rise > prob_thread:
                sig = 1
            else:
                sig = 0
        elif self.sig_meth == 2:
            '''转换真实信号'''
            sig = df['y_real'].iloc[0]
            if sig == 0:
                sig = -1
            elif sig == 1:
                sig = 0
            else:
                sig = 1
        elif self.sig_meth == 3:
            sig = df['y_pred'].iloc[0]
            if sig == 0:
                sig = -1
        return sig
    
    def close_pos(self, bar):
        if self.pos > 0:
            self.sell(bar.close_price, abs(self.pos))

        elif self.pos < 0:
            self.cover(bar.close_price, abs(self.pos))

    def on_n_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        am = self.amn
        am.update_bar(bar)
        # if not am.inited:
        #     return
        
        # try:
        #     self.is_macon_change = self.df_ma[self.df_ma['date']==str(bar.datetime.date())]['is_change'].iloc[0]
        # except:
        #     self.is_macon_change = 0
        #     # print('出错了', bar.datetime)

        # if self.is_macon_change:
        #     self.is_count_change = False
        #     self.sig = 0
        #     self.count_sig = 0
        #     self.close_pos(bar)
        
        # else:
        # print('hour:', bar.datetime)
        # input()
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.pre_sig = self.sig
            self.sig = self.transfrom_sig(self.df_pred[self.df_pred['datetime'] == bdt])
        except:
            pass
            self.count += 1
            # print(bdt, self.sig, self.count)

        # if self.count_sig == self.step_n:
        #     try:
        #         self.sig = self.transfrom_sig(self.df_pred[self.df_pred['datetime'] == bdt])
        #         success = 1
        #     except:
        #         pass
        #         # self.close_pos(bar)
        #         success = 0
        #     self.count_sig = 1 if success else self.step_n
        # else:
        #     self.count_sig += 1
        if not self.amn.inited:
            return 
        self.save_info(bar)

    def save_info(self, bar: BarData):
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.res_dic['datetime'].append(bdt)
        self.res_dic['close'].append(bar.close_price)
        self.res_dic['pos'].append(self.pos)

        # print('1小时：', bar.datetime, self.beign_trade)
        # try:
        if self.beign_trade == 1:       # 计算当前交易价格和交易时间
            trade_price = self.long_pre_price
            trade_time = self.trade_datetime
        elif self.beign_trade == -1:
            trade_price = self.short_pre_price
            trade_time = self.trade_datetime
        else:
            trade_price = 0
            trade_time = 0
        # except:
        #     print(self.amn.inited, '00000')
        #     trade_price = 0
        #     input()

        self.beign_trade = 0    # 重置beign_trade
        self.res_dic['trade_price'].append(trade_price)

        if self.first:
            self.first = 0
            signal = self.sig
            self.res_dic['profit'].append(0)
        else:
            signal = self.sig if self.sig == -np.sign(self.pos) else 0
            if trade_price == 0:
                self.res_dic['profit'].append((self.res_dic['close'][-1]-self.res_dic['close'][-2])*self.pos*self.size)
            else:
                self.res_dic['profit'].append(((self.res_dic['close'][-1]-self.res_dic['trade_price'][-1])*self.pos+\
                (self.res_dic['trade_price'][-1] - self.res_dic['close'][-2])*(-self.pos))*self.size)
        
        self.res_dic['trade_time'].append(trade_time)
        self.res_dic['signal'].append(signal)
        cost = 2*(self.rate*bar.close_price + 0.5*self.pricetick)*self.size if signal != 0 else 0
        self.res_dic['cost'].append(cost)
        # print('dic_len:', len(self.res_dic['datetime']), self.res_dic['datetime'][-1])


    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        # if bar.datetime.minute == 59:
        #     print(bar.datetime)
        #     input()

        self.bgn.update_bar(bar)

        am = self.am
        am.update_bar(bar)
        
        if not am.inited:
            return
        
        # print('minute:', bar.datetime, self.sig, self.pos)
        
        if self.pos == 0:
            if self.sig == 1:
                self.buy(bar.close_price, self.hand)
                # self.beign_trade = 1
            elif self.sig == -1:
                self.short(bar.close_price, self.hand)
                # self.beign_trade = -1
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                self.short(bar.close_price, self.hand)
                # self.beign_trade = -1
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                self.buy(bar.close_price, self.hand)
                # self.beign_trade = 1

        # 后一分钟挂单
        # order_price = bar.open_price
        # if self.pos == 0:
        #     if self.sig == 1:
        #         if self.beign_trade == 1:
        #             self.buy(order_price, self.hand)
        #         else:
        #             self.beign_trade = 1
        #             # print('beign_trade', self.beign_trade)
        #     elif self.sig == -1:
        #         if self.beign_trade == -1:
        #             self.short(order_price, self.hand)
        #         else:
        #             self.beign_trade = -1
        #             # print('beign_trade', self.beign_trade)
        # elif self.pos > 0:
        #     if self.sig == -1:
        #         if self.beign_trade == -1:
        #             self.sell(order_price, abs(self.pos))
        #             self.short(order_price, self.hand)
        #         else:
        #             self.beign_trade = -1
        #             # print('beign_trade', self.beign_trade)
        # elif self.pos < 0:
        #     if self.sig == 1:
        #         if self.beign_trade == 1:
        #             self.cover(bar.close_price, abs(self.pos))
        #             self.buy(bar.close_price, self.hand)
        #         else:
        #             self.beign_trade = 1
        #             # print('beign_trade', self.beign_trade)
        
        self.put_event()

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        # print(trade)
        if trade.direction == Direction.LONG:
            self.long_pre_price = trade.price
            # print(self.am.datetime[-1], self.long_pre_price)
        else:
            self.short_pre_price = trade.price
            # print(self.am.datetime[-1], self.short_pre_price)

        self.beign_trade = np.sign(self.pos)
        self.trade_datetime = trade.datetime

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
