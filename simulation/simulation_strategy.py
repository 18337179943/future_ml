from datetime import timedelta, datetime, time
import imp
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
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
from vnpy.trader.constant import Interval, Offset
import random
from backtesting.ml_strategy_adj import MLStrategy
from simulation.base_strategy import BaseStrategy, ArrayManager_m
import talib as tb
sys.path.append("..")


class SimulationStrategy(BaseStrategy):
    """
    trade_type
    0、前n根最高最低价止盈止损
    1、前n根涨跌幅超过k%, 当前只做反转
    2、如果后n根收盘价比自己前一根收盘价高, 则止损
    3、后n根k线短时间超跌时止损
    4、后n根k线如果亏损时加仓
    5、后n根k线收盘价最大值没有超过开盘价时止损
    6、如果后第n根k线为浮亏时止损
    {'PP': 4, 'AL': 7, 'AP': 8}
    """
    author = "用Python的交易员"

    hand = 1    # 手数
    symbol_name = 'IF'  # 合约名称
    step_n = 12     # 预测步数
    y_pred = 'y_pred'
    pa = 'test_datas_sharp_ratio_30m_5m'
    sig_meth = 0
    win_n = 1
    rate = 0
    size = 10
    pricetick = 0
    atr_n = 10
    atr_profit_dev = 2.0
    atr_loss_dev = 1.0
    trend_n = 4
    revers_n = 2
    signal_thread1 = 0.003
    signal_thread2 = 0.01
    max_pos = 2
    contract = 'RB2205'

    is_leverage = 0

    trade_type = 0

    # trade_mod
    stop_loss_n = 5 

    # trade_mod1
    acc_n = 4
    atr_dev = 1.0

    # trade_mod2

    # trade_mod3
    # atr_mod3_dev = 1.0

    # trade_mod4

    # trade_mod5

    # trade_mod9
    loss_n = 2

    # trade_mod10
    is_win = 1

    # trade_mod12
    init_balance = 1_000_000

    # trade_mod13
    profit_rate_li = []

    atr_n_mod13 = 5
    atr_dev_mod13 = 0.5

    # trade_mod14
    pos_mod = 0
    atr_n_mod14 = 4
    atr_dev_mod14 = 0.4
    last_direction = 0

    # trade_mod15
    open_time = time(9, 0)
    atr_n_mod15 = 5
    atr_dev_mod15 = 0.5

    # trade_mod16
    limit_n = 2

    # trade_mod19
    atr_n_mod19 = 4
    atr_dev_mod19 = 0.2

    # trade_mod21
    stop_loss_rate = 0.03

    # trade_mod23
    atr_n_mod23 = 4
    atr_dev_mod23 = 0.4

    # trade_mod24
    atr_n_mod24 = 4
    atr_dev_mod24 = 7
    atr_rate_mod24 = 0.5
    atr_dynamic = 0
    profit_rate_loss = 50


    sig = 0         # 当前信号
    stop_profit_price = 0
    stop_loss_price = 0
    pre_sig = 0

    # val_trade_mod2
    count_open = 0
    pre_pos_sig = 0

    # val_trade_mod13
    accumulate_val = 0

    # val_trade_mod14
    is_big_win = 0

    # val_trade_mod16
    uplimitprice = 0
    downlimitprice = 0
    up_stop_price = 0
    down_stop_price = 0
    count_limit = 0
    occur_limit = 0

    # val_trade_mod20
    hand_mod20 = 0
    start_price = 0

    # val_trade_mod22
    highest_price = 0

    # val_trade_mod23
    stop_loss_price_mod23 = 0

    # val_trade_mod24
    stop_loss_price_mod24 = 0
    
    long_pre_price = 0
    short_pre_price = 0

    need_open_pos = 0
    open_price = 0

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
        "pricetick",
        "atr_n",
        "atr_dev",
        "atr_profit_dev",
        "atr_loss_dev",
        "trend_n",
        "revers_n",
        "signal_thread1",
        "signal_thread2",
        "stop_loss_n",
        "trade_type",
        "max_pos",
        "loss_n",
        "is_win",
        "is_leverage",
        "init_balance",
        "contract",
        "profit_rate_li",
        "pos_mod",
        "atr_n_mod13",
        "atr_dev_mod13",
        "atr_n_mod15",
        "atr_dev_mod15",
        "atr_n_mod14",
        "atr_dev_mod14",
        "limit_n",
        "atr_n_mod19",
        "atr_dev_mod19",
        "stop_loss_rate",
        "atr_n_mod23",
        "atr_dev_mod23",
        "atr_n_mod24",
        "atr_dev_mod24",
        "atr_rate_mod24",
        "atr_dynamic",
        "profit_rate_loss"
    ]
    variables = [
        "sig",
        "stop_profit_price",
        "stop_loss_price",
        "pre_sig",
        "count_open",
        "pre_pos_sig"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        it = Interval.MINUTE if self.win_n != 1 else Interval.HOUR
        self.bg = BarGenerator(self.on_bar)
        self.bgn = BarGenerator(self.on_bar, self.win_n, self.on_n_bar, interval=it)
        self.am = ArrayManager_m(100)
        # max_n = np.max([self.atr_n*2+1, self.trend_n, self.acc_n])
        self.amn = ArrayManager_m(self.atr_n_mod24+1)
        self.is_macon_change = 0
        self.count = 0
        self.res_dic = {'datetime': [], 'open': [], 'high': [], 'low': [], 'close': [], 'trade_price': [], 'trade_time': [], 'pred_sig': [],
                        'signal': [], 'pos': [], 'profit': [], 'cost': [], 'signal_class': [],
                        'trend_pct': [], 'revers_pct': [], 'balance': []}  # 'trend_rise_prob': [], 'revers_rise_prob': [], 'trend_rise_n': [], 'revers_rise_n': [], 
        self.first = 1
        self.trade_res = self.reset_trade_res()  # 0方向 1价格 2仓位 3时间
        self.is_init = 1
        self.trigger_type = ''
        self.having_night = False
        self.need_close_pos = 0
        self.ignore_signal = 0
        self.loss_count = 0
        self.loss_pos = 0

        if self.is_leverage == 0:
            self.hand = int(1_000_000 * self.hand * 0.15 / 300_000)
            self.hand_mod20 = self.hand
        
    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

#         df = pd.read_csv('{pa_prefix}/datas/maincon.csv') 
#         self.df_ma = df[df['symbol'] == self.symbol_name].copy()     # 获取品种主力合约数据
        
        # self.df_ma = df[df['is_change']==1]     # 判断合约换月
        # self.df_ma['date'] = self.df_ma['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # self.df_pred = pd.read_csv(f'{pa_prefix}/datas/data_set/{self.symbol_name}/{self.pa}.csv')
        self.df_contract = pd.read_csv(f'{pa_prefix}/datas/data_1m/{self.symbol_name}/{self.contract}.csv')
        
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
    
    def signal_class(self, sig):
        '''信号分类'''
        # 使用红色k线和绿色k线的比例来判断趋势
        # diff_close_open = self.amn.close[-self.trend_n:] - self.amn.open[-self.trend_n:]
        # trend_rise_n = np.sum(np.where(diff_close_open > 0, 1, 0))
        # revers_rise_n = np.sum(np.where(diff_close_open[-self.revers_n:] > 0, 1, 0))
        # trend_rise_prob = trend_rise_n / self.trend_n
        # revers_rise_prob = revers_rise_n / self.revers_n
        trend_pct = (self.amn.close[-1] - self.amn.close[-self.trend_n]) / self.amn.close[-self.trend_n]
        revers_pct = (self.amn.close[-1] - self.amn.close[-self.revers_n]) / self.amn.close[-self.revers_n]
        # trend_pct_thread = 0.0
        sign_trend, sign_revers, sign_sig = np.sign(trend_pct), np.sign(revers_pct), np.sign(sig)
        
        if sig != 0:
            if abs(revers_pct) > self.signal_thread1:
                if sign_sig == sign_revers:
                    signal_class = 'trend'
                else:
                    if sign_trend == sign_sig and abs(revers_pct) > self.signal_thread2:
                        signal_class = 'adjust'
                    else:
                        signal_class = 'revers'
            else:
                signal_class = 'other'
        else:
            signal_class = ''
        
        # if sig != 0:
        #     if abs(trend_pct) > trend_pct_thread:
        #         if sign_trend == sign_sig:      # 趋势和信号一致的情况下
        #             if sign_trend == sign_revers:  
        #                 signal_class = 'trend'
        #             else:
        #                 signal_class = 'adjust'
        #         else:
        #             signal_class = 'revers'
        #     else:
        #         signal_class = 'trend'
        # else:
        #     signal_class = ''

        return signal_class, trend_pct, revers_pct
    
    def on_n_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        if bar.volume == 0:
            return
        am = self.amn
        am.update_bar(bar)
        # if not am.inited:
        #     return

        # atr_value = self.amn.atr(self.atr_n)
        # ma_value = self.amn.sma(self.atr_n)
        # self.stop_profit_price = self.atr_profit_dev*atr_value
        
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.pre_sig = self.sig
            self.sig = self.transfrom_sig(self.df_pred[self.df_pred['datetime'] == bdt])
        except:
            # print(self.symbol_name, bdt)
            self.count += 1
            # print(bdt, self.sig, self.count)

        if not self.amn.inited:
            return 

        if self.pos != 0:
            self.count_open += 1
        # if self.is_init:
        #     print('start:', bar.datetime)
        #     self.is_init = 0
        self.save_info(bar)

        if self.atr_dynamic: # trade_mod24
            self.stop_loss_price_mod24 = self.atr_dev_mod24 * self.m_atr(self.atr_n_mod24, self.atr_rate_mod24)
    
    def save_info(self, bar: BarData):
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.res_dic['datetime'].append(bdt)
        self.res_dic['open'].append(bar.open_price)
        self.res_dic['high'].append(bar.high_price)
        self.res_dic['low'].append(bar.low_price)
        self.res_dic['close'].append(bar.close_price)
        self.res_dic['pos'].append(self.pos)
        self.res_dic['pred_sig'].append(self.sig)
        
        # one_hand_cost = (self.rate*bar.close_price + 0.5*self.pricetick)*self.size
        one_hand_cost = (self.rate*bar.close_price)*self.size
        
        if self.first:
            self.first = 0
            signal = self.sig
            self.res_dic['profit'].append(0)
            self.res_dic['trade_price'].append(0)
            self.res_dic['trade_time'].append(0)
            cost = one_hand_cost
            balance = self.init_balance
            self.start_price = bar.close_price
        else:
            # if self.trade_type == 8:
            #     ignore_signal = self._caculate_signal_ignore()
            # else:
            #     ignore_signal = False

            # if self.trade_type == 9 and self.loss_count >= self.loss_n:
            #     lc = 1
            # else:
            #     lc = 0

            # if lc==0:
            #     if self.trigger_type == 'already_stop_loss' and self.sig == -self.pre_pos_sig:
            #         signal = self.sig
            #     elif (self.sig == -np.sign(self.pos) and not ignore_signal) or (self.pos == 0 and self.trigger_type != 'already_stop_loss'):
            #         signal = self.sig
            #     else:
            #         signal = 0
            # else:
            #     signal = 0

            # elif self.trade_type == 1:
            #     signal = self.sig if self.sig == -np.sign(self.pos) or self.pos == 0 else 0
            # else:
            #     print('not done yet.')
            if self.trade_type == 0:
                signal = self.sig if self.sig == -np.sign(self.pos) or self.pos == 0 else 0
            elif self.trade_type == 19 or self.trade_type == 21 or self.trade_type == 24:
                if self.trigger_type == 'already_stop_loss' and self.sig == -self.pre_pos_sig:
                    signal = self.sig
                elif self.sig == -np.sign(self.pos) or (self.pos == 0 and self.trigger_type != 'already_stop_loss'):
                    signal = self.sig
                else:
                    signal = 0
            else:
                if bar.datetime.time() == time(14, 0):
                    if self.pos == 0:
                        self.loss_count = 0
                    self.occur_limit = 0
                self.need_open_pos = (self.loss_count < self.loss_n and self.hand > 0)*1#  and not self.occur_limit

                if self.need_open_pos and (self.sig == -np.sign(self.pos) or self.pos == 0):
                    signal = self.sig
                else:
                    signal = 0

            profit_i, cost = 0, 0
            price_li = self.trade_res['price'].copy()
            price_li.insert(0, self.res_dic['close'][-2]), price_li.append(bar.close_price)
            pos_li = self.trade_res['pos'].copy()
            pos_li.insert(0, self.res_dic['pos'][-2])

            for i in range(len(price_li)-1):
                profit_i += (price_li[i+1] - price_li[i])*pos_li[i]*self.size
            
            self.res_dic['profit'].append(profit_i)
            self.res_dic['trade_price'].append(self.trade_res['price'])
            self.res_dic['trade_time'].append(self.trade_res['datetime'])
            
            cost = one_hand_cost*len(self.trade_res['price'])*self.hand
            balance = self.res_dic['balance'][-1] + profit_i - cost
            
        self.res_dic['signal'].append(signal)
        self.res_dic['cost'].append(cost)
        self.res_dic['balance'].append(balance)
        # signal_class, trend_pct, revers_pct = self.signal_class(signal)   # 信号分类
        signal_class, trend_pct, revers_pct = 0, 0, 0
        self.res_dic['signal_class'].append(signal_class)
        # self.res_dic['trend_rise_prob'].append(trend_rise_prob)
        # self.res_dic['revers_rise_prob'].append(revers_rise_prob)
        # self.res_dic['trend_rise_n'].append(trend_rise_prob*self.trend_n)
        # self.res_dic['revers_rise_n'].append(trend_rise_prob*self.trend_n)
        self.res_dic['trend_pct'].append(trend_pct)
        self.res_dic['revers_pct'].append(revers_pct)
        self.trade_res = self.reset_trade_res()

        if self.trade_type == 12 or self.trade_type == 100 or self.trade_type == 102 or self.trade_type == 101: # or self.trade_type == 0:
            self.hand = int(balance / self.size / bar.close_price)
        
        if self.trade_type == 20:
            max_hand = int(3 * balance / (self.size * bar.close_price))
            self.hand_mod20 = int(min(((balance / self.init_balance - 1)*10 + 1), 3) * balance / (self.size * self.start_price))
            self.hand_mod20 = min(max(self.hand, self.hand_mod20), max_hand)

    def reset_trade_res(self):
        '''每隔一小时重设交易记录'''
        trade_res = {'price': [], 'pos': [], 'datetime': []}
        return trade_res
        
    def trade_mod0(self, bar: BarData):
        '''毛信号'''
        if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29) or bar.datetime.time() == time(14, 59) or \
            bar.datetime.time() == time(15, 0) or bar.datetime.time() == time(23, 0): # or bar.datetime.time() == time(14, 59)
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or
        else:

            self.hand = 1_000_000 / bar.close_price / self.size
            if self.pos == 0:
                if self.hand > 0:
                    if self.sig == 1:
                        self.buy(bar.close_price, self.hand)
                    elif self.sig == -1:
                        self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    if self.hand > 0:
                        self.short(bar.close_price, self.hand)
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    if self.hand > 0:
                        self.buy(bar.close_price, self.hand)
    
    def trade_mod1(self, bar: BarData):
        '''前n根涨跌幅超过k%, 当前只做反转'''
        diff_price = self.amn.close[-self.acc_n:] - self.amn.open[-self.acc_n:]
        price_rate = self.amn.close[-1] - self.amn.close[-self.acc_n]
        atr_v = tb.ATR(self.amn.high[:-self.acc_n], self.amn.low[:-self.acc_n], self.amn.close[:-self.acc_n], self.atr_n)[-1]
        thread_rate = self.atr_dev*atr_v
        rise_rate = np.sum(np.where(diff_price>0, 1, 0)) / self.acc_n
        is_rise_trend = 1 if rise_rate >= 0.8 and price_rate > thread_rate else 0
        is_down_trend = 1 if rise_rate <= 0.2 and price_rate < -thread_rate else 0

        if self.pos == 0:
            if self.sig == 1 and is_rise_trend == 0:
                self.buy(bar.close_price, self.hand)
            elif self.sig == -1 and is_down_trend == 0:
                self.short(bar.close_price, self.hand)

        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if is_down_trend == 0:
                    self.short(bar.close_price, self.hand)

        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if is_rise_trend == 0:
                    self.buy(bar.close_price, self.hand)
    
    def trade_mod2(self, bar: BarData):
        '''如果后n根收盘价比自己前一根收盘价高, 则止损'''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig != self.sig) or self.trigger_type != 'already_stop_loss':
            # if self.trigger_type == 'already_stop_loss':
            #     print(self.pre_pos_sig, self.sig, self.pos)
            #     input()
            # else:
            #     print(self.pre_pos_sig, self.sig, self.pos, '---')
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    self.short(bar.close_price, self.hand)
                else:
                    if self.count_open == self.stop_loss_n:
                        for i in range(1, self.stop_loss_n+1):
                            need_stop_loss = (self.amn.close[-i]<self.amn.close[-i-1])*1
                        if need_stop_loss:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'
                    # self.beign_trade = -1
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)
                else:
                    if self.count_open == self.stop_loss_n:
                        for i in range(1, self.stop_loss_n+1):
                            need_stop_loss = (self.amn.close[-i]>self.amn.close[-i-1])*1
                        if need_stop_loss:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'

    def trade_mod3(self, bar: BarData):
        '''后n根k线短时间超跌时止损'''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig != self.sig) or self.trigger_type == '':
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
                else:
                    if self.count_open <= self.stop_loss_n:
                        if self.amn.close[-self.count_open] - bar.close_price > self.amn.atr(self.atr_n)*self.atr_dev:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'
                    # self.beign_trade = -1
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)
                else:
                    if self.count_open <= self.stop_loss_n:
                        if bar.close_price - self.amn.close[-self.count_open] > self.amn.atr(self.atr_n)*self.atr_dev:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'

    def trade_mod4(self, bar: BarData):
        '''后n根k线如果亏损时加仓
        适用pp, 参数: stop_loss_n=5'''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
            last_close_price = self.amn.close[-1]
            pred_sig_val = abs(np.sum(self.res_dic['pred_sig'][-self.stop_loss_n:]))
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    self.short(bar.close_price, self.hand)
                else:
                    if self.count_open == self.stop_loss_n and self.long_pre_price > last_close_price and pred_sig_val == self.stop_loss_n:
                        self.sell(bar.close_price, abs(self.pos))
                        self.trigger_type = 'long_stop_loss'
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)
                else:
                    if self.count_open == self.stop_loss_n and self.short_pre_price < last_close_price and pred_sig_val == self.stop_loss_n:
                        self.cover(bar.close_price, abs(self.pos))
                        self.trigger_type = 'short_stop_loss'

    def trade_mod5(self, bar: BarData):
        '''后n根k线收盘价最大值没有超过开仓价时止损'''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig != self.sig) or self.trigger_type == '':
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    self.short(bar.close_price, self.hand)
                else:
                    if self.count_open == self.stop_loss_n and np.max(self.amn.close[-self.stop_loss_n:]) < bar.close_price:
                        self.sell(bar.close_price, abs(self.pos))
                        self.trigger_type = 'long_stop_loss'
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)
                else:
                    if self.count_open == self.stop_loss_n and np.min(self.amn.close[-self.stop_loss_n:]) > bar.close_price:
                        self.cover(bar.close_price, abs(self.pos))
                        self.trigger_type = 'short_stop_loss'

    def trade_mod6(self, bar: BarData):
        '''如果后第n根k线为浮亏时止损'''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
            last_close_price = self.amn.close[-1]
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    self.short(bar.close_price, self.hand)
                else:
                    if self.count_open >= self.stop_loss_n and self.long_pre_price > last_close_price:
                        self.sell(bar.close_price, abs(self.pos))
                        self.trigger_type = 'long_stop_loss'
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)
                else:
                    if self.count_open >= self.stop_loss_n and self.short_pre_price < last_close_price:
                        self.cover(bar.close_price, abs(self.pos))
                        self.trigger_type = 'short_stop_loss'

    def trade_mod7(self, bar:BarData):
        '''前n个信号相同, n+1的信号不同, 如果亏损则平仓，盈利则持仓
        example1: 1 0 0 0 0 1 0 则亏损平仓
        example2: 1 1 1 1 1 -1 1 则把-1仓位平掉不开仓直到下一个-1
        适用于AL, 参数stop_loss_n = 4
        '''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
            last_close_price = self.amn.close[-1]
            pred_sig_li = self.res_dic['pred_sig']
            # diff_price = self.amn.close - self.amn.open
            if len(pred_sig_li) >= self.stop_loss_n+3:
                pred_sig_count = abs(np.sum(pred_sig_li[-self.stop_loss_n-1:-1]))
                need_stop_loss = (pred_sig_count == 0 or pred_sig_count == self.stop_loss_n) and pred_sig_li[-1] != pred_sig_li[-2] \
                    and (pred_sig_li[-1] == 0 or pred_sig_li[-2] == 0)
                self.need_close_pos = pred_sig_li[-1] == pred_sig_li[-3] and pred_sig_li[-2] == -pred_sig_li[-1] and \
                                        abs(np.sum(pred_sig_li[-self.stop_loss_n-2:-2])) == self.stop_loss_n and not self.having_night
                                        # self.amn.datetime[-1].time() >= time(13, 0) and 
                # self.need_close_pos = 0
            else:
                need_stop_loss = 0
                self.need_close_pos = 0
            
            # need_open_pos = self.having_night or (not self.having_night and bar.datetime.time() < time(14, 30))

            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)

            elif self.pos > 0:
                if self.need_close_pos:
                    self.sell(bar.close_price, abs(self.pos))
                    self.trigger_type = 'long_stop_loss'
                else:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        self.short(bar.close_price, self.hand)
                    else:
                        if (need_stop_loss and self.long_pre_price > last_close_price): # or \
                            # (bar.open_price > self.amn.high[-1] and diff_price[-1]<0 and diff_price[-2]<0):
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'
                    
            elif self.pos < 0:
                if self.need_close_pos:
                    self.cover(bar.close_price, abs(self.pos))
                    self.trigger_type = 'short_stop_loss'
                else:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        self.buy(bar.close_price, self.hand)
                    else:
                        if (need_stop_loss and self.short_pre_price < last_close_price): # or \
                            # (bar.open_price < self.amn.low[-1] and diff_price[-1]>0 and diff_price[-2]>0):
                            if self.short_pre_price < last_close_price:
                                self.cover(bar.close_price, abs(self.pos))
                                self.trigger_type = 'short_stop_loss'
            
    def trade_mod8(self, bar:BarData):
        '''如果 1 -1 1 1, 则多持有一个周期，如 1 -1 -1 1
        如果持仓出现两次9点都是盈利的，则平仓
        适用于AP，参数 stop_loss_n
        '''
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
            # last_close_price = self.amn.close[-1]
            self._caculate_signal_ignore()
                
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if not self.ignore_signal or self.amn.close[-1] > self.amn.close[-2]:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        self.short(bar.close_price, self.hand)
            elif self.pos < 0:
                if not self.ignore_signal or self.amn.close[-1] < self.amn.close[-2]:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        self.buy(bar.close_price, self.hand)
                    
    def _caculate_signal_ignore(self):
        pred_sig_li = self.res_dic['pred_sig']
        pos_li = self.res_dic['pos']
        if len(pred_sig_li) >= self.stop_loss_n+3:
            self.ignore_signal = pred_sig_li[-1] == -pred_sig_li[-2] and pred_sig_li[-3] == pred_sig_li[-1] and \
                                    pred_sig_li[-1] != 0 and pred_sig_li[-1] == -pos_li[-1] and pred_sig_li[-2] == -pos_li[-2]
        else:
            self.ignore_signal = False
        return self.ignore_signal

    def trade_mod9(self, bar:BarData):
        '''如果开仓亏损次数超过n次, 当天就不再开仓'''
        if bar.datetime.time() == time(14, 59) and self.pos == 0:
            self.loss_count = 0

        if self.pos == 0:
            if (self.loss_count < self.loss_n and self.loss_pos == -self.sig) or self.loss_pos==0:
                self.loss_pos = 0
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                # if self.loss_count < self.loss_n:
                #     self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                # if self.loss_count < self.loss_n:
                #     self.buy(bar.close_price, self.hand)
        return 

    def trade_mod10(self, bar:BarData):
        '''如果开仓亏损次数超过n次, 当天就不再开仓'''
        if bar.datetime.time() == time(14, 59) and self.pos == 0:
            self.loss_count = 0

        # if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
        if self.pos == 0:
            if self.loss_count < self.loss_n:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                    # if len(self.profit_rate_li) > 4:
                    #     print(self.profit_rate_li[-4:], self.loss_count, bar.close_price, 'buy', bar.datetime)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
                    # if len(self.profit_rate_li) > 4:
                    #     print(self.profit_rate_li[-4:], self.loss_count, bar.close_price, 'short', bar.datetime)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                # if len(self.profit_rate_li) > 4:
                #     print(self.profit_rate_li[-4:], self.loss_count, bar.close_price - self.long_pre_price, 'sell', bar.datetime)
                # self.loss_count += 1 if bar.close_price < self.long_pre_price else 0
                # if self.loss_count < self.loss_n:
                #     self.short(bar.close_price, self.hand)
                #     if len(self.profit_rate_li) > 4:
                #         print(self.profit_rate_li[-4:], self.loss_count, bar.close_price, 'short', bar.datetime)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                # if len(self.profit_rate_li) > 4:
                #     print(self.profit_rate_li[-4:], self.loss_count, self.short_pre_price - bar.close_price, 'cover', bar.datetime)
                # self.loss_count += 1 if bar.close_price > self.short_pre_price else 0
                # if self.loss_count < self.loss_n:
                #     self.buy(bar.close_price, self.hand)
                #     if len(self.profit_rate_li) > 4:
                #         print(self.profit_rate_li[-4:], self.loss_count, bar.close_price, 'buy', bar.datetime)
                # input()
        return 

    def trade_mod11(self, bar:BarData):
        '''如果开仓后n根k线后出现阳包阴并且持空头，平仓'''
        
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
            last_close_price = self.amn.close[-1]
            diff_co = self.amn.close - self.amn.open
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    self.short(bar.close_price, self.hand)
                else:
                    if self.count_open >= self.stop_loss_n and diff_co[-1] < 0 and diff_co[-2] > 0 and abs(diff_co[-1]) > abs(diff_co[-2]):
                        self.sell(bar.close_price, abs(self.pos))
                        self.trigger_type = 'long_stop_loss'
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)
                else:
                    if self.count_open >= self.stop_loss_n and diff_co[-1] > 0 and diff_co[-2] < 0 and abs(diff_co[-1]) > abs(diff_co[-2]):
                        self.cover(bar.close_price, abs(self.pos))
                        self.trigger_type = 'short_stop_loss'

    def trade_mod12(self, bar: BarData):
        '''仓位按照总资金变动'''
        if self.pos == 0:
            if self.hand > 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                    # self.beign_trade = 1
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
                    # self.beign_trade = -1
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if self.hand > 0:
                    self.short(bar.close_price, self.hand)
                # self.beign_trade = -1
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if self.hand > 0:
                    self.buy(bar.close_price, self.hand)
    
    def trade_mod13(self, bar:BarData):
        '''如果开仓属于小亏，则连续亏损到达k倍的atr，则当天不再开仓，若盈利达到k倍atr则重新计算
        '''
        
        if bar.datetime.time() == time(14, 59) and self.pos == 0:
            self.accumulate_val = 0

        # def need_open_pos(self, is_sell):
        #     if is_sell:
        #         trade_return = self.am.close[-1] - self.long_pre_price
        #     else:
        #         trade_return = self.short_pre_price - self.am.close[-1]
        #     self.accumulate_val += trade_return
        #     atr_thread = self.atr_n_mod13*self.amn.atr(self.atr_n_mod13)
        #     if self.accumulate_val > atr_thread or trade_return > atr_thread:
        #         self.accumulate_val = 0
        
        need_open_pos = self.accumulate_val > -self.atr_dev_mod13*self.amn.atr(self.atr_n_mod13)
        # print(self.contract, need_open_pos, self.accumulate_val, self.amn.atr(self.atr_n_mod13), self.atr_dev_mod13)
        # if self.big_loss_count == self.big_loss_n:
        #     print('大亏损------')
        # if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
        if self.pos == 0:
            if need_open_pos:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                # elif self.sig == -1:
                #     self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                # if need_open_pos:
                #     self.buy(bar.close_price, self.hand)
        return 

    def trade_mod14(self, bar:BarData):
        '''如果开仓属于大赚，则平仓/半仓
        '''
        def get_atr_thread(self):
            return self.atr_dev_mod14*self.amn.atr(self.atr_n_mod14)

        if self.pos == 0:
            if self.sig == 1:
                if self.is_big_win:
                    if self.pos_mod == 1:
                        self.is_big_win = 0
                        self.buy(bar.close_price, self.hand // 2)
                    elif self.sig == self.last_direction:
                        self.is_big_win = 0
                        self.buy(bar.close_price, self.hand)
                else:
                    self.buy(bar.close_price, self.hand)

            elif self.sig == -1:
                if self.is_big_win:
                    if self.pos_mod == 1:
                        self.is_big_win = 0
                        self.short(bar.close_price, self.hand // 2)
                    elif self.sig == self.last_direction:
                        self.is_big_win = 0
                        self.short(bar.close_price, self.hand)
                else:
                    self.short(bar.close_price, self.hand)

        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if bar.close_price - self.long_pre_price > get_atr_thread(self):
                    if self.pos_mod == 1:
                        self.short(bar.close_price, self.hand // 2)
                else:
                    self.short(bar.close_price, self.hand)
                    
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if self.short_pre_price - bar.close_price > get_atr_thread(self):
                    if self.pos_mod == 1:
                        self.buy(bar.close_price, self.hand // 2)
                else:
                    self.buy(bar.close_price, self.hand)

    def trade_mod15(self, bar: BarData):
        '''如果14点前小赚/亏损，则平仓'''
        if bar.datetime.time() >= time(14, 50):
            if self.open_time < time(14, 0) or self.open_time > time(20, 30):
                atr_thread = self.atr_n_mod15*self.atr_dev_mod15
                if self.pos > 0:
                    if bar.close_price - self.long_pre_price < atr_thread:
                        self.sell(bar.close_price, abs(self.pos))
                elif self.pos < 0:
                    if self.short_pre_price - bar.close_price < atr_thread:
                        self.cover(bar.close_price, abs(self.pos))
        else:
            if self.pos == 0:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    self.short(bar.close_price, self.hand)
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    self.buy(bar.close_price, self.hand)

    def trade_mod16(self, bar: BarData):
        '''快到涨跌停时如果持有反仓则平仓，过limit_n后再开仓'''
        almost_to_uplimit = bar.close_price >= self.up_stop_price
        almost_to_downlimit = bar.close_price <= self.down_stop_price
        if almost_to_uplimit or almost_to_downlimit:
            self.occur_limit = 1 
        
        if bar.datetime.time() == time(14, 59):
            if self.occur_limit:
                self.occur_limit = 0
                self.count_limit = 0 if self.limit_n == 0 else 1
            elif 0 < self.count_limit < self.limit_n:
                self.count_limit += 1
            else:
                self.count_limit = 0

        if self.count_limit == 0:
            if self.pos == 0:
                if not self.occur_limit:
                    if self.sig == 1:
                        self.buy(bar.close_price, self.hand)
                    elif self.sig == -1:
                        self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if almost_to_downlimit:
                    self.sell(bar.close_price, abs(self.pos))
                else:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        if not self.occur_limit: 
                            self.short(bar.close_price, self.hand)
            elif self.pos < 0:
                if almost_to_uplimit:
                    self.cover(bar.close_price, abs(self.pos))
                else:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        if not self.occur_limit: 
                            self.buy(bar.close_price, self.hand)
        else:
            if self.pos > 0 and (almost_to_downlimit or self.sig == -1):
                self.sell(bar.close_price, abs(self.pos))
            elif self.pos < 0 and (almost_to_uplimit or self.sig == 1):
                self.cover(bar.close_price, abs(self.pos))

    def trade_mod17(self, bar:BarData):
        '''如果信号亏损次数超过n次,再开一仓后平仓, 当天就不再开仓'''
        if bar.datetime.time() == time(14, 59) and self.pos == 0:
            self.loss_count = 0

        # if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
        if self.pos == 0:
            if self.loss_count < self.loss_n:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if self.loss_count < self.loss_n:
                    self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if self.loss_count < self.loss_n:
                    self.buy(bar.close_price, self.hand)
        return 

    def trade_mod18(self, bar:BarData):
        '''如果开仓属于小亏，则连续亏损到达k倍的atr，则再开一笔，如果累计高于止损值，则继续开仓，否则当天不再开仓，若盈利达到k倍atr则重新计算
        '''
        
        if bar.datetime.time() == time(14, 59) and self.pos == 0:
            self.accumulate_val = 0

        need_open_pos = self.accumulate_val > -self.atr_dev_mod13*self.amn.atr(self.atr_n_mod13)
        # print(self.contract, need_open_pos, self.accumulate_val, self.amn.atr(self.atr_n_mod13), self.atr_dev_mod13)
        # if self.big_loss_count == self.big_loss_n:
        #     print('大亏损------')
        # if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
        if self.pos == 0:
            if need_open_pos:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.buy(bar.close_price, self.hand)

    def trade_mod19(self, bar:BarData):
        '''k倍atr止损'''
        if bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 30):
            return 
        if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
            if self.pos == 0:
                if self.hand > 0:
                    if self.sig == 1:
                        self.buy(bar.close_price, self.hand)
                    elif self.sig == -1:
                        self.short(bar.close_price, self.hand)
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    if self.hand > 0:
                        self.short(bar.close_price, self.hand)
                else:
                    if self.long_pre_price - self.amn.close[-1] > self.atr_dev_mod19*self.amn.atr(self.atr_n_mod19):
                        self.sell(bar.close_price, abs(self.pos))
                        self.trigger_type = 'long_stop_loss'

            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    if self.hand > 0:
                        self.buy(bar.close_price, self.hand)
                else:
                    if self.amn.close[-1] - self.short_pre_price > self.atr_dev_mod19*self.amn.atr(self.atr_n_mod19):
                        self.cover(bar.close_price, abs(self.pos))
                        self.trigger_type = 'short_stop_loss'

    def trade_mod20(self, bar: BarData):
        '''通过调整仓位，盈利则杠杆提高，亏损则杠杆降低，最高三倍杠杆'''
        if bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or 
        else:
            if self.pos == 0:
                if self.hand_mod20 > 0:
                    if self.sig == 1:
                        self.buy(bar.close_price, self.hand_mod20)
                        # self.beign_trade = 1
                    elif self.sig == -1:
                        self.short(bar.close_price, self.hand_mod20)
                        # self.beign_trade = -1
            elif self.pos > 0:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    if self.hand_mod20 > 0:
                        self.short(bar.close_price, self.hand_mod20)
                    # self.beign_trade = -1
            elif self.pos < 0:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    if self.hand_mod20 > 0:
                        self.buy(bar.close_price, self.hand_mod20)

    def trade_mod21(self, bar: BarData):
        '''亏损3%止损'''
        if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or 
        else:
            if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
                if self.pos == 0:
                    if self.hand > 0:
                        if self.sig == 1:
                            self.buy(bar.close_price, self.hand)
                        elif self.sig == -1:
                            self.short(bar.close_price, self.hand)
                elif self.pos > 0:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.short(bar.close_price, self.hand)
                    else:
                        if (self.long_pre_price - bar.close_price) / self.long_pre_price > self.stop_loss_rate:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'

                elif self.pos < 0:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.buy(bar.close_price, self.hand)
                    else:
                        if (bar.close_price - self.short_pre_price) / self.short_pre_price > self.stop_loss_rate:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'
    
    def trade_mod22(self, bar: BarData):
        '''最高点亏损3%止损'''
        if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or 
        else:
            if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
                if self.pos == 0:
                    if self.hand > 0:
                        if self.sig == 1:
                            self.buy(bar.close_price, self.hand)
                        elif self.sig == -1:
                            self.short(bar.close_price, self.hand)
                elif self.pos > 0:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.short(bar.close_price, self.hand)
                    else:
                        # 止损
                        self.highest_price = max(self.highest_price, bar.high_price)
                        if (self.highest_price - bar.close_price) / self.highest_price > self.stop_loss_rate:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'

                elif self.pos < 0:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.buy(bar.close_price, self.hand)
                    else:
                        self.highest_price = min(self.highest_price, bar.low_price)
                        if (bar.close_price - self.highest_price) / self.highest_price > self.stop_loss_rate:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'

    def trade_mod23(self, bar: BarData):
        '''最高点亏损n倍atr止损'''
        if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or 
        else:
            if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
                if self.pos == 0:
                    if self.hand > 0:
                        if self.sig == 1:
                            self.buy(bar.close_price, self.hand)
                        elif self.sig == -1:
                            self.short(bar.close_price, self.hand)
                elif self.pos > 0:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.short(bar.close_price, self.hand)
                    else:
                        # 止损
                        self.highest_price = max(self.highest_price, bar.high_price)
                        if (self.highest_price - bar.close_price) > self.stop_loss_price_mod23:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'

                elif self.pos < 0:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.buy(bar.close_price, self.hand)
                    else:
                        self.highest_price = min(self.highest_price, bar.low_price)
                        if (bar.close_price - self.highest_price) > self.stop_loss_price_mod23:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'
            
    def trade_mod24(self, bar: BarData):
        '''最高点亏损n倍atr止损, atr取前k个'''
        if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or 
        else:
            if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
                if self.pos == 0:
                    if self.hand > 0:
                        if self.sig == 1:
                            self.buy(bar.close_price, self.hand)
                        elif self.sig == -1:
                            self.short(bar.close_price, self.hand)
                elif self.pos > 0:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.short(bar.close_price, self.hand)
                    else:
                        # 止损
                        self.highest_price = max(self.highest_price, bar.high_price)
                        if (((self.highest_price - bar.close_price) > self.stop_loss_price_mod24) or \
                            ((bar.close_price - self.long_pre_price) > self.profit_rate_loss * self.stop_loss_price_mod24)) and self.stop_loss_price_mod24 != 0:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'

                elif self.pos < 0:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.buy(bar.close_price, self.hand)
                    else:
                        self.highest_price = min(self.highest_price, bar.low_price)
                        if ((bar.close_price - self.highest_price) > self.stop_loss_price_mod24 or \
                            (self.short_pre_price - bar.close_price) > self.profit_rate_loss * self.stop_loss_price_mod24) and self.stop_loss_price_mod24 != 0:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'
            
    def trade_mod25(self, bar: BarData):
        '''最高点亏损n倍atr止损, atr取前k个'''
        if bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 14) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29):
            return  # bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or 
        else:
            if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
                if self.pos == 0:
                    if self.hand > 0:
                        if self.sig == 1:
                            self.buy(bar.close_price, self.hand)
                        elif self.sig == -1:
                            self.short(bar.close_price, self.hand)
                elif self.pos > 0:
                    if self.sig == -1:
                        self.sell(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.short(bar.close_price, self.hand)
                    else:
                        # 止损
                        self.highest_price = max(self.highest_price, bar.high_price)
                        if (self.highest_price - bar.close_price) > self.stop_loss_price_mod24 and self.stop_loss_price_mod24 != 0:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'
                        elif (bar.close_price - self.highest_price) > self.stop_loss_price_mod24 and self.stop_loss_price_mod24 != 0:
                            self.sell(bar.close_price, abs(self.pos))
                            self.trigger_type = 'long_stop_loss'

                elif self.pos < 0:
                    if self.sig == 1:
                        self.cover(bar.close_price, abs(self.pos))
                        if self.hand > 0:
                            self.buy(bar.close_price, self.hand)
                    else:
                        self.highest_price = min(self.highest_price, bar.low_price)
                        if (bar.close_price - self.highest_price) > self.stop_loss_price_mod24 and self.stop_loss_price_mod24 != 0:
                            self.cover(bar.close_price, abs(self.pos))
                            self.trigger_type = 'short_stop_loss'

    def trade_mod100(self, bar:BarData):
        '''组合策略12 16 17'''
        if bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0) or bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(10, 30):
            return 
        almost_to_uplimit = bar.close_price >= self.up_stop_price
        almost_to_downlimit = bar.close_price <= self.down_stop_price
        if almost_to_uplimit or almost_to_downlimit:
            self.occur_limit = 1 

        if bar.datetime.time() == time(14, 59):
            if self.pos == 0:
                self.loss_count = 0
            self.occur_limit = 0
        self.need_open_pos = (self.loss_count < self.loss_n and self.hand > 0)*1 and not self.occur_limit

        if self.pos == 0:
            if self.need_open_pos:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            if almost_to_downlimit:
                self.sell(bar.close_price, abs(self.pos))
            else:
                if self.sig == -1:
                    self.sell(bar.close_price, abs(self.pos))
                    # if need_open_pos:
                    #     self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            if almost_to_uplimit:
                self.cover(bar.close_price, abs(self.pos))
            else:
                if self.sig == 1:
                    self.cover(bar.close_price, abs(self.pos))
                    # if need_open_pos:
                    #     self.buy(bar.close_price, self.hand)
        return 
    
    def trade_mod101(self, bar:BarData):
        '''组合策略12 16 18'''
        # almost_to_uplimit = bar.close_price >= self.up_stop_price
        # almost_to_downlimit = bar.close_price <= self.down_stop_price
        # if almost_to_uplimit or almost_to_downlimit:
        #     self.occur_limit = 1 

        # need_open_pos = self.loss_count < self.loss_n # and not self.occur_limit
        if bar.datetime.time() == time(14, 59) and self.pos == 0:
            self.accumulate_val = 0

        need_open_pos = self.accumulate_val > -self.atr_dev_mod13*self.amn.atr(self.atr_n_mod13)
        # print(self.contract, need_open_pos, self.accumulate_val, self.amn.atr(self.atr_n_mod13), self.atr_dev_mod13)
        # if self.big_loss_count == self.big_loss_n:
        #     print('大亏损------')
        # if (self.trigger_type == 'already_stop_loss' and self.pre_pos_sig == -self.sig) or self.trigger_type != 'already_stop_loss':
        if self.pos == 0:
            if need_open_pos:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.buy(bar.close_price, self.hand)

    def trade_mod102(self, bar:BarData):
        '''组合策略12 16 17 18'''
        # almost_to_uplimit = bar.close_price >= self.up_stop_price
        # almost_to_downlimit = bar.close_price <= self.down_stop_price
        # if almost_to_uplimit or almost_to_downlimit:
        #     self.occur_limit = 1 

        # need_open_pos = self.loss_count < self.loss_n # and not self.occur_limit
        if bar.datetime.time() == time(14, 59):
            if self.pos == 0:
                self.loss_count = 0
                self.accumulate_val = 0
            self.occur_limit = 0
        need_open_pos = self.accumulate_val > -self.atr_dev_mod13*self.amn.atr(self.atr_n_mod13) and self.loss_count < self.loss_n

        if self.pos == 0:
            if need_open_pos:
                if self.sig == 1:
                    self.buy(bar.close_price, self.hand)
                elif self.sig == -1:
                    self.short(bar.close_price, self.hand)
        elif self.pos > 0:
            # if almost_to_downlimit:
            #     self.sell(bar.close_price, abs(self.pos))
            # else:
            if self.sig == -1:
                self.sell(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.short(bar.close_price, self.hand)
        elif self.pos < 0:
            # if almost_to_uplimit:
            #     self.cover(bar.close_price, abs(self.pos))
            # else:
            if self.sig == 1:
                self.cover(bar.close_price, abs(self.pos))
                if need_open_pos:
                    self.buy(bar.close_price, self.hand)

    def _get_up_dwon_limit_price(self, bar: BarData):
        '''获取当天的涨跌停价格'''
        if bar.datetime.time() == time(9, 0) or bar.datetime.time() == time(21, 0):
            bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            try:
                self.uplimitprice, self.downlimitprice = self.df_contract[self.df_contract['datetime'] == bdt][['UpperLimitPrice', 'LowerLimitPrice']].values[0]
            except:
                # pass
                self.uplimitprice, self.downlimitprice = bar.close_price*1.2, bar.close_price*0.8
            pct_price_val = 0.005*bar.close_price
            self.up_stop_price, self.down_stop_price = self.uplimitprice - pct_price_val, self.downlimitprice + pct_price_val
                # print(self.contract, bar.datetime, '--------')
        elif self.uplimitprice == 0:
            self.uplimitprice, self.downlimitprice = bar.close_price*1.2, bar.close_price*0.8
        
        return self.uplimitprice, self.downlimitprice

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        # if bar.datetime.minute == 59:
        #     print(bar.datetime)
        #     input()
        bar_time = bar.datetime.time()

        if time(15, 0) <= bar_time < time(20, 40):
            return 
        elif bar_time == time(8, 59):
            self.open_price = bar.open_price
            self.open_volume = bar.volume
            self.open_turnover = bar.turnover
            return 
        elif bar_time == time(20, 59):
            self.having_night = 1
            self.open_price = bar.open_price
            self.open_volume = bar.volume
            self.open_turnover = bar.turnover
            return 
        elif bar_time == time(23, 0):  # 上期所的除外
            return 
        else:
            if bar_time == time(14, 59):
                self.having_night = 0
                self.open_price = 0
            
            elif bar_time == time(9, 0) and not self.having_night and self.open_price != 0:
                bar.open_price = self.open_price
                bar.volume = bar.volume + self.open_volume
                bar.turnover = bar.turnover + self.open_turnover
            
            elif bar_time == time(21, 0) and self.open_price != 0:
                bar.open_price = self.open_price
                bar.volume = bar.volume + self.open_volume
                bar.turnover = bar.turnover + self.open_turnover
            
            self.bgn.update_bar(bar)

            # self.bgn.update_bar(bar)

            am = self.am
            am.update_bar(bar)
        
            if not am.inited:
                return
                
            if bar.datetime.hour == 21 and bar.datetime.minute == 0:
                self.having_night = True  # 有夜盘
            elif bar.datetime.time() >= time(14, 30) and bar.datetime.time() < time(21, 0):
                self.having_night = False

            self._get_up_dwon_limit_price(bar)

            getattr(self, f'trade_mod{self.trade_type}')(bar)
        
        self.put_event()
        
    def _caculate_loss_count(self, trade: TradeData):
        '''计算损失笔数'''
        if not self.amn.inited:
            return 

        # trade_price_li = self.trade_record_dic['trade_price']
        profit_rate_li = self.profit_rate_li

        trade_price = trade.price
        # print(trade.offset)

        is_trade_close = (trade.offset == Offset.CLOSE or trade.offset == Offset.CLOSETODAY or trade.offset == Offset.CLOSEYESTERDAY)
        # print(trade.direction, trade.offset, trade.datetime, trade_price)

        if is_trade_close:
            if trade.direction == Direction.LONG:
                trade_return  = (self.short_pre_price - trade_price)
                profit_rate_li.append(trade_return / self.short_pre_price) 
            else:
                trade_return = (trade_price - self.long_pre_price)
                profit_rate_li.append(trade_return / self.long_pre_price)

            # trade_mod9-12
            if self.trade_type == 9 or self.trade_type == 10 or self.trade_type == 11:
                if self.is_win:
                    if trade.direction == Direction.LONG:
                        self.loss_count = self.loss_count + 1 if trade_price < self.short_pre_price else 0
                    else:
                        self.loss_count = self.loss_count + 1 if trade_price > self.long_pre_price else 0
                else:
                    if trade.direction == Direction.LONG:
                        self.loss_count = self.loss_count + 1 if trade_price > self.short_pre_price else 0
                        # print(self.loss_count, self.profit_rate_li[-1], profit_rate_li[-1], trade_price > self.short_pre_price)
                    else:
                        self.loss_count = self.loss_count + 1 if trade_price < self.long_pre_price else 0
                        # print(self.loss_count, self.profit_rate_li[-1], profit_rate_li[-1], trade_price < self.long_pre_price)

            # arr_profit_rate = np.array(profit_rate_li)
            # return_rate = profit_rate_li[-1]

            elif self.trade_type == 13 or self.trade_type == 18 or self.trade_type == 102 or self.trade_type == 101:
                # trade_mod13 18
                self.accumulate_val += trade_return
                atr_thread = self.atr_n_mod13*self.amn.atr(self.atr_n_mod13)
                if self.accumulate_val > atr_thread or trade_return > atr_thread:
                    self.accumulate_val = 0

            elif self.trade_type == 14:
                # trade_mod14
                atr_thread = self.atr_dev_mod14*self.amn.atr(self.atr_n_mod14)
                self.is_big_win = 1 if trade_return > atr_thread else 0
                self.last_direction = -1 if trade.direction == Direction.LONG else 1
                # if self.is_big_win:
                #     print(self.contract, trade.datetime, trade_return, atr_thread)

        else:
            self.open_time = trade.datetime.time()

        if self.trade_type == 17 or self.trade_type == 100 or self.trade_type == 102:
            # trade_mod17
            if self.long_pre_price != 0 or self.short_pre_price != 0:
                if self.loss_count < self.loss_n:
                    if trade.direction == Direction.LONG:
                        self.loss_count = self.loss_count + 1 if trade.price > self.short_pre_price else 0
                    elif trade.direction == Direction.SHORT:
                        self.loss_count = self.loss_count + 1 if trade.price < self.long_pre_price else 0

        if self.loss_count == self.loss_n:
            self.loss_pos = -self.pre_pos_sig

    def m_atr(self, n, k):
        '''tr取前百分之k求平均, mod24'''
        # print('amn', self.amn.close)
        # print('----------', len(s))
        high_low = self.amn.high[1:] - self.amn.low[1:]
        pre_close_high = np.abs(self.amn.close[:-1] - self.amn.high[1:])
        pre_close_low = np.abs(self.amn.close[:-1] - self.amn.low[1:])
        tr_arr = np.max((high_low, pre_close_high, pre_close_low), axis=0)[-n:]

        atr_n = int(n * k)
        atr = np.mean(np.sort(tr_arr)[-atr_n:])
        return atr

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """

        self._caculate_loss_count(trade)    # 计算累计损失笔数

        if trade.direction == Direction.LONG:
            self.long_pre_price = trade.price
            self.pre_pos_sig = -1
            # print(self.am.datetime[-1], self.long_pre_price)
        else:
            self.short_pre_price = trade.price
            self.pre_pos_sig = 1
            # print(self.am.datetime[-1], self.short_pre_price)

        if self.need_close_pos:
            self.pre_pos_sig = -self.pre_pos_sig

        if self.trigger_type == 'long_stop_loss' or self.trigger_type == 'short_stop_loss':
            self.trigger_type = 'already_stop_loss'
            # self.sig = 0

        elif self.trigger_type == 'already_stop_loss':
            self.trigger_type = ''

        self.count_open = 0

        self.highest_price = trade.price

        # trade_mod23
        self.stop_loss_price_mod23 = self.atr_dev_mod23 * self.amn.atr(self.atr_n_mod23)

        # trade_mod24
        if not self.atr_dynamic:
            self.stop_loss_price_mod24 = self.atr_dev_mod24 * self.m_atr(self.atr_n_mod24, self.atr_rate_mod24)
        
        self.trade_res['price'].append(trade.price)
        self.trade_res['pos'].append(self.pos)
        self.trade_res['datetime'].append(trade.datetime.strftime('%Y-%m-%d %H:%M:%S'))
        # if self.first == 0:
        #     print(self.contract, self.pos, self.res_dic['balance'][-1])
        self.put_event()
