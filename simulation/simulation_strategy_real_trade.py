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

    sep_hand_n = 3


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
    pre_pos = 0

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
        "profit_rate_loss",
        "sep_hand_n"
    ]
    variables = [
        "sig",
        "stop_profit_price",
        "stop_loss_price",
        "pre_sig",
        "count_open",
        "pre_pos_sig",
        "pre_pos"
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
        self.start_open_price = 0
        # self.start_open = 0
        self.start_hand_li = []
        # self.last_hand = 0      # 前一次开仓手数
        self.long_serise = 0  # 多头排列个数
        self.pre_hand = 0

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
            pass
            self.count += 1
            print(bdt, self.sig, self.count)

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

    def init_hand_li(self, bar: BarData):
        ''''''
        self.hand = int(300_000 / bar.close_price / self.size)
        if self.hand < self.sep_hand_n:
            hand_li = [1]*self.hand + [0]*(self.sep_hand_n-self.hand)
        else:
            left_hand = self.hand % self.sep_hand_n
            single_hand = self.hand // self.sep_hand_n
            hand_li = [single_hand]*self.sep_hand_n
            if left_hand:
                for i in range(left_hand):
                    hand_li[i] = hand_li[i] + 1
        return hand_li

    def set_long_serise(self):
        '''判断多头排列'''
        long_serise = 0
        diff_price_arr = self.am.close[-10:] - self.am.open[-10:]
        diff_price_arr = diff_price_arr[diff_price_arr!=0]
        if diff_price_arr[-1] > 0:
            long_serise = 1
            if diff_price_arr[-2] > 0:
                long_serise = 2
        else:
            long_serise = -1
            if diff_price_arr[-2] < 0:
                long_serise = -2
        return long_serise
    
    def get_open_hand(self):
        '''获取开仓手数'''
        if self.long_serise == -1:
            left_hand = self.pre_hand - abs(self.pos - self.pre_pos)
            ind = self.start_hand_li.index(np.max(self.start_hand_li))
            self.start_hand_li[ind] = self.start_hand_li[ind] + left_hand
            self.pre_pos = self.pos
            open_hand = np.min(self.start_hand_li)



    def trade_mod0(self, bar: BarData):
        '''毛信号'''
        self.hand = int(300_000 / bar.close_price / self.size)
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
        '''毛信号'''
        if bar.datetime.time() == time(14, 59) or bar.datetime.time() == time(21, 59) or bar.datetime.time() == time(22, 59) or bar.datetime.time() == time(23, 29) or \
            bar.datetime.time() == time(0, 59) or bar.datetime.time() == time(2, 29) or bar.datetime.time() == time(9, 59) or bar.datetime.time() == time(10, 59) or \
            bar.datetime.time() == time(11, 29) or bar.datetime.time() == time(13, 59): # or bar.datetime.time() == time(14, 59)
            if self.pos * self.sig < 0 or (self.pos == 0 and self.sig != 0):
                self.start_open_price = bar.close_price
                # self.start_open = 1
                self.start_hand_li = self.init_hand_li(bar)
                self.long_serise = self.set_long_serise()
        else:
            if self.pos * self.sig >= 0 and abs(self.pos) == self.hand:
                # self.start_open = 0
                self.long_serise = 0

        hand = 2

        if self.pos == 0:
            if self.hand > 0:
                if self.sig == 1:
                    self.buy(bar.close_price, hand)
                elif self.sig == -1:
                    self.short(bar.close_price, hand)
        elif self.pos > 0:
            if self.sig == -1:
                self.sell(bar.close_price, min(hand, abs(self.pos)))
                if self.hand > 0:
                    self.short(bar.close_price, hand)
        elif self.pos < 0:
            if self.sig == 1:
                self.cover(bar.close_price, min(hand, abs(self.pos)))
                if self.hand > 0:
                    self.buy(bar.close_price, hand)
                

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
            self.have_night = 1
            self.open_price = bar.open_price
            self.open_volume = bar.volume
            self.open_turnover = bar.turnover
            return 
        else:
            if bar_time == time(14, 59):
                self.have_night = 0
                self.open_price = 0
            
            elif bar_time == (9, 0) and not self.have_night and self.open_price != 0:
                bar.open_price = self.open_price
                bar.volume = bar.volume + self.open_volume
                bar.turnover = bar.turnover + self.open_turnover
            
            elif bar_time == (21, 0) and self.open_price != 0:
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
