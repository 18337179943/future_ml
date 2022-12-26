from vnpy.app.cta_backtester import CtaBacktesterApp

# 以动态绑定的方式替换回测引擎类
import vnpy.app.cta_backtester.engine as backtester_engine
from .backtesting import BacktestingEngine
backtester_engine.BacktestingEngine = BacktestingEngine

import vnpy.app.cta_strategy.backtesting as cta_backtesting
cta_backtesting.BacktestingEngine = BacktestingEngine

import vnpy.app.cta_backtester.ui.widget as backtester_widget
from .widget import StatisticsMonitor
backtester_widget.StatisticsMonitor = StatisticsMonitor

from .widget import OptimizationResultMonitor
backtester_widget.OptimizationResultMonitor = OptimizationResultMonitor