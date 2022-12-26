import numpy as np
from pandas import DataFrame
from datetime import date
from collections import defaultdict

from vnpy.trader.object import TradeData
from vnpy.trader.constant import Direction
from vnpy.app.cta_strategy.backtesting import BacktestingEngine as OriginalEngine


class TradeResult:

    def __init__(self, size: int, rate: float, slippage: float):
        """"""
        self.size = size
        self.rate = rate
        self.slippage = slippage

        self.pos = 0

        self.long_cost = 0      # 多头成交成本
        self.short_cost = 0     # 空头成交成本
        self.trade_volume = 0   # 成交量
        self.trade_amount = 0   # 成交额

        self.total_slippage = 0     # 总滑点
        self.total_commission = 0   # 总手续费
        self.total_pnl = 0          # 总盈亏
        self.net_pnl = 0            # 净营亏

        self.open_time = None       # 开仓时间
        self.close_time = None      # 平仓时间

    def update_trade(self, trade: TradeData):
        """"""
        if not self.open_time:
            self.open_time = trade.datetime

        self.trade_volume += trade.volume
        self.trade_amount += trade.volume * trade.price

        if trade.direction == Direction.LONG:
            self.pos += trade.volume
            self.long_cost += trade.volume * trade.price * self.size
        else:
            self.pos -= trade.volume
            self.short_cost += trade.volume * trade.price * self.size

        if not self.pos:
            self.close_time = trade.datetime

            self.total_slippage = self.slippage * self.size * self.trade_volume
            self.total_commission = self.rate * self.size * self.trade_amount
            self.total_pnl = self.short_cost - self.long_cost
            self.net_pnl = self.total_pnl - self.total_slippage - self.total_commission


class BacktestingEngine(OriginalEngine):
    """"""

    def calculate_result(self):
        """"""
        # 逐笔对冲统计计算
        self.trade_results = []
        trade_result = TradeResult(self.size, self.rate, self.slippage)

        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("成交记录为空，无法计算")
            return

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)

            # 更新逐笔对冲记录
            trade_result.update_trade(trade)
            if not trade_result.pos:
                self.trade_results.append(trade_result)
                trade_result = TradeResult(self.size, self.rate, self.slippage)

        # Calculate daily result by iteration.
        pre_close = 0
        start_pos = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate,
                self.slippage,
                self.inverse
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("开始计算策略统计指标")

        # print("--------------------修改后的calculate_statistics--------------------")

        # 逐笔对冲统计
        profit_count = 0
        loss_count = 0
        total_count = len(self.trade_results)
        total_trade_profit = 0
        total_trade_loss = 0

        for trade_result in self.trade_results:
            if trade_result.net_pnl >= 0:
                profit_count += 1
                total_trade_profit += trade_result.net_pnl
            else:
                loss_count += 1
                total_trade_loss += trade_result.net_pnl

        if profit_count:
            average_trade_profit = total_trade_profit / profit_count
        else:
            average_trade_profit = 0

        if loss_count:
            average_trade_loss = total_trade_loss / loss_count
        else:
            average_trade_loss = 0

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            total_days = 0
            profit_days = 0
            loss_days = 0
            end_balance = 0
            max_drawdown = 0
            max_ddpercent = 0
            max_drawdown_duration = 0
            total_net_pnl = 0
            daily_net_pnl = 0
            total_commission = 0
            daily_commission = 0
            total_slippage = 0
            daily_slippage = 0
            total_turnover = 0
            daily_turnover = 0
            total_trade_count = 0
            daily_trade_count = 0
            total_return = 0
            annual_return = 0
            daily_return = 0
            return_std = 0
            sharpe_ratio = 0
            return_drawdown_ratio = 0
            profit_days_ratio = 0
            profit_loss_ratio = 0
            commission_pnl_ratio = 0
            pnl_drawdown_ratio = 0
            pnl_std_ratio = 0
            total_count = 0
            profit_count = 0
            loss_count = 0
            average_trade_profit = 0
            average_trade_loss = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * 240
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                sharpe_ratio = daily_return / return_std * np.sqrt(240)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_return / max_ddpercent

            # 扩展内容
            profit_days_ratio = profit_days / total_days * 100     # 胜率

            total_profit = df[df["net_pnl"] > 0]["net_pnl"].sum()
            total_loss = df[df["net_pnl"] < 0]["net_pnl"].sum()
            average_profit = total_profit / profit_days
            average_loss = total_loss / loss_days
            profit_loss_ratio = abs(average_profit / average_loss)  # 盈亏比

            commission_pnl_ratio = total_commission / total_net_pnl  # 俑损比

            annual_pnl = total_net_pnl / total_days * 240
            pnl_drawdown_ratio = abs(annual_pnl / max_drawdown)     # 年化回撤比

            daily_pnl = df["net_pnl"].mean()
            pnl_std = df["net_pnl"].std()
            pnl_std_ratio = daily_pnl / pnl_std * 100               # 离散系数

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
            # 新增字段
            "profit_days_ratio": profit_days_ratio,
            "profit_loss_ratio": profit_loss_ratio,
            "commission_pnl_ratio": commission_pnl_ratio,
            "pnl_drawdown_ratio": pnl_drawdown_ratio,
            "pnl_std_ratio": pnl_std_ratio,

            "total_count": total_count,
            "profit_count": profit_count,
            "loss_count": loss_count,
            "average_trade_profit": average_trade_profit,
            "average_trade_loss": average_trade_loss
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        return statistics
