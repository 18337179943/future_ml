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
import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
from vnpy.trader.constant import Interval
import random
from backtesting.ml_strategy_adj import MLStrategy
sys.path.append("..")


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


class BaseStrategy(MLStrategy):
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
    atr_n =  10
    atr_profit_dev = 2.0
    atr_loss_dev = 1.0
    trend_n = 4
    revers_n = 2
    signal_thread1 = 0.003
    signal_thread2 = 0.01

    sig = 0         # 当前信号
    stop_profit_price = 0
    stop_loss_price = 0
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
        "pricetick",
        "atr_n",
        "atr_profit_dev",
        "atr_loss_dev",
        "trend_n",
        "revers_n",
        "signal_thread1",
        "signal_thread2"
    ]
    variables = [
        "sig",
        "stop_profit_price",
        "stop_loss_price",
        "pre_sig"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        it = Interval.MINUTE if self.win_n != 1 else Interval.HOUR
        self.bg = BarGenerator(self.on_bar)
        self.bgn = BarGenerator(self.on_bar, self.win_n, self.on_n_bar, interval=it)
        self.am = ArrayManager_m(100)
        self.amn = ArrayManager_m(self.trend_n+1)
        self.is_macon_change = 0
        self.count = 0
        self.res_dic = {'datetime': [], 'open': [], 'high': [], 'low': [], 'close': [], 'trade_price': [], 'trade_time': [], 'pred_sig': [],
                        'signal': [], 'pos': [], 'profit': [], 'cost': [], 'signal_class': [],
                        'trend_pct': [], 'revers_pct': []}  # 'trend_rise_prob': [], 'revers_rise_prob': [], 'trend_rise_n': [], 'revers_rise_n': [], 
        self.first = 1
        self.trade_res = self.reset_trade_res()  # 0方向 1价格 2仓位 3时间
        self.is_init = 1

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
        if isinstance(self.y_pred, str):
            # self.df_pred = pd.read_csv(f'{pa_prefix}/datas/predict/{self.symbol_name}/{self.y_pred}.csv')    # 获取预测信号
            self.df_pred = pd.read_csv(f'{self.y_pred}')    # 获取预测信号
            if 'y' in self.df_pred.columns.to_list():
                self.df_pred = self.df_pred.rename(columns={'y': 'y_pred'})
        else:
            self.df_pred = self.y_pred.copy()
            self.df_pred['datetime'] = self.df_pred['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
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

        atr_value = self.amn.atr(self.atr_n)
        # ma_value = self.amn.sma(self.atr_n)
        self.stop_profit_price = self.atr_profit_dev*atr_value
        self.stop_loss_price = self.atr_loss_dev*atr_value
        
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.pre_sig = self.sig
            self.sig = self.transfrom_sig(self.df_pred[self.df_pred['datetime'] == bdt])
        except:
            pass
            self.count += 1
            # print(bdt, self.sig, self.count)

        if not self.amn.inited:
            return 
        # if self.is_init:
        #     print('start:', bar.datetime)
        #     self.is_init = 0
        self.save_info(bar)
    
    def save_info(self, bar: BarData):
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.res_dic['datetime'].append(bdt)
        self.res_dic['open'].append(bar.open_price)
        self.res_dic['high'].append(bar.high_price)
        self.res_dic['low'].append(bar.low_price)
        self.res_dic['close'].append(bar.close_price)
        self.res_dic['pos'].append(self.pos)
        self.res_dic['pred_sig'].append(self.sig)
        
        one_hand_cost = (self.rate*bar.close_price + 0.5*self.pricetick)*self.size
        
        if self.first:
            # print('first--', bar.datetime, self.stop_profit_price, self.stop_loss_price, bar.close_price, self.amn.atr(self.atr_n))
            # print(self.amn.close)
            self.first = 0
            signal = self.sig
            self.res_dic['profit'].append(0)
            self.res_dic['trade_price'].append(0)
            self.res_dic['trade_time'].append(0)
            cost = one_hand_cost
        else:
            signal = self.sig if self.sig == -np.sign(self.pos) or self.pos == 0 else 0
            profit_i = 0
            price_li = self.trade_res['price'].copy()
            price_li.insert(0, self.res_dic['close'][-2]), price_li.append(bar.close_price)
            pos_li = self.trade_res['pos'].copy()
            pos_li.insert(0, self.res_dic['pos'][-2])

            for i in range(len(price_li)-1):
                profit_i += (price_li[i+1] - price_li[i])*pos_li[i]*self.size
            
            self.res_dic['profit'].append(profit_i)
            self.res_dic['trade_price'].append(self.trade_res['price'])
            self.res_dic['trade_time'].append(self.trade_res['datetime'])
            
            cost = one_hand_cost*len(self.trade_res['price'])
            
        self.res_dic['signal'].append(signal)
        self.res_dic['cost'].append(cost)
        signal_class, trend_pct, revers_pct = self.signal_class(signal)
        self.res_dic['signal_class'].append(signal_class)
        # self.res_dic['trend_rise_prob'].append(trend_rise_prob)
        # self.res_dic['revers_rise_prob'].append(revers_rise_prob)
        # self.res_dic['trend_rise_n'].append(trend_rise_prob*self.trend_n)
        # self.res_dic['revers_rise_n'].append(trend_rise_prob*self.trend_n)
        self.res_dic['trend_pct'].append(trend_pct)
        self.res_dic['revers_pct'].append(revers_pct)
        self.trade_res = self.reset_trade_res()

        
    def reset_trade_res(self):
        '''每隔一小时重设交易记录'''
        trade_res = {'price': [], 'pos': [], 'datetime': []}
        return trade_res
        
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
        
        self.trade_res['price'].append(trade.price)
        self.trade_res['pos'].append(self.pos)
        self.trade_res['datetime'].append(trade.datetime.strftime('%Y-%m-%d %H:%M:%S'))
        self.put_event()
