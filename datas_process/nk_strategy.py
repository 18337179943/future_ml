from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
from vnpy.trader.constant import Interval
from datetime import time



class NkLine(CtaTemplate):
    """"""

    author = "用Python的交易员"

    win_n = 5
    open_price = 0
    open_volume = 0
    open_turnover = 0
    have_night = 0

    parameters = [
        'win_n'
    ]
    variables = [
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        if self.win_n == 60:
            interval = Interval.HOUR 
            self.win_n = 1
        else:
            interval = Interval.MINUTE

        self.bg = BarGenerator(self.on_bar, self.win_n, self.on_n_bar, interval=interval)
        self.am = ArrayManager()
        self.amn = ArrayManager()
        self.li_res = []
        self.col = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

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

    def on_n_bar(self, bar: BarData):

        self.amn.update_bar(bar)
        if bar.volume != 0:
            self.li_res.append([bar.datetime, bar.open_price, bar.high_price, bar.low_price, bar.close_price, bar.volume, bar.turnover])

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        am = self.am
        am.update_bar(bar)

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
            
            elif bar_time == time(9, 0) and not self.have_night and self.open_price != 0:
                bar.open_price = self.open_price
                bar.volume = bar.volume + self.open_volume
                bar.turnover = bar.turnover + self.open_turnover
            
            elif bar_time == time(21, 0) and self.open_price != 0 and self.have_night:
                bar.open_price = self.open_price
                bar.volume = bar.volume + self.open_volume
                bar.turnover = bar.turnover + self.open_turnover

            self.bg.update_bar(bar)

        if not am.inited:
            return

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
