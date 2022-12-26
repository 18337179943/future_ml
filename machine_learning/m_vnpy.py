from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from datetime import datetime





class CTARunBacktest():
    def __init__(self, strategy, vt_symbol, start, end, rate, slippage, size, pricetick, interval='1m', params={}) -> None:
        self.engine = BacktestingEngine()
        self.engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=interval,
            start=start,
            end=end,
            rate=rate,
            slippage=slippage,
            size=size,
            pricetick=pricetick,
            capital=1_000_000,
        )
        self.engine.add_strategy(strategy, params)
        
    def run_backtest(self):
        self.engine.load_data()
        self.engine.run_backtesting()
        df = self.engine.calculate_result()
        self.engine.calculate_statistics()
        return df
    
    def show_chart(self):
        self.engine.show_chart()

    def optimize(self):
        '''优化还没有写好'''
        setting = OptimizationSetting()
        setting.set_target("sharpe_ratio")
        setting.add_parameter("atr_length", 25, 27, 1)
        setting.add_parameter("atr_ma_length", 10, 30, 10)

        self.engine.run_ga_optimization(setting)

        self.engine.run_bf_optimization(setting)

