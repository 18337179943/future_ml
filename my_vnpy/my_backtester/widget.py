import csv

from vnpy.app.cta_backtester.ui.widget import StatisticsMonitor as OriginalStatiticsMonitor
from vnpy.app.cta_backtester.ui.widget import OptimizationResultMonitor as OriginalOptimizationResultMonitor
from vnpy.trader.ui import QtCore, QtWidgets


class StatisticsMonitor(OriginalStatiticsMonitor):
    """"""

    KEY_NAME_MAP = {
        "start_date": "首个交易日",
        "end_date": "最后交易日",

        "total_days": "总交易日",
        "profit_days": "盈利交易日",
        "loss_days": "亏损交易日",

        "capital": "起始资金",
        "end_balance": "结束资金",

        "total_return": "总收益率",
        "annual_return": "年化收益",
        "max_drawdown": "最大回撤",
        "max_ddpercent": "百分比最大回撤",

        "total_net_pnl": "总盈亏",
        "total_commission": "总手续费",
        "total_slippage": "总滑点",
        "total_turnover": "总成交额",
        "total_trade_count": "总成交笔数",

        "daily_net_pnl": "日均盈亏",
        "daily_commission": "日均手续费",
        "daily_slippage": "日均滑点",
        "daily_turnover": "日均成交额",
        "daily_trade_count": "日均成交笔数",

        "daily_return": "日均收益率",
        "return_std": "收益标准差",
        "sharpe_ratio": "夏普比率",
        "return_drawdown_ratio": "收益回撤比",

        "profit_days_ratio": "胜率（交易日）",
        "profit_loss_ratio": "盈亏比（交易日）",
        "commission_pnl_ratio": "佣损比",
        "pnl_drawdown_ratio": "收益回撤比",
        "pnl_std_ratio": "标准离差率",

        "total_count": "交易结果次数",
        "profit_count": "盈利结果次数",
        "loss_count": "亏损结果次数",
        "average_trade_profit": "平均盈利金额",
        "average_trade_loss": "平均亏损金额"
    }

    def set_data(self, data: dict):
        """"""
        data["capital"] = f"{data['capital']:,.2f}"
        data["end_balance"] = f"{data['end_balance']:,.2f}"
        data["total_return"] = f"{data['total_return']:,.2f}%"
        data["annual_return"] = f"{data['annual_return']:,.2f}%"
        data["max_drawdown"] = f"{data['max_drawdown']:,.2f}"
        data["max_ddpercent"] = f"{data['max_ddpercent']:,.2f}%"
        data["total_net_pnl"] = f"{data['total_net_pnl']:,.2f}"
        data["total_commission"] = f"{data['total_commission']:,.2f}"
        data["total_slippage"] = f"{data['total_slippage']:,.2f}"
        data["total_turnover"] = f"{data['total_turnover']:,.2f}"
        data["daily_net_pnl"] = f"{data['daily_net_pnl']:,.2f}"
        data["daily_commission"] = f"{data['daily_commission']:,.2f}"
        data["daily_slippage"] = f"{data['daily_slippage']:,.2f}"
        data["daily_turnover"] = f"{data['daily_turnover']:,.2f}"
        data["daily_return"] = f"{data['daily_return']:,.2f}%"
        data["return_std"] = f"{data['return_std']:,.2f}%"
        data["sharpe_ratio"] = f"{data['sharpe_ratio']:,.2f}"
        data["return_drawdown_ratio"] = f"{data['return_drawdown_ratio']:,.2f}"

        data["profit_days_ratio"] = f"{data['profit_days_ratio']:,.2f}%"
        data["profit_loss_ratio"] = f"{data['profit_loss_ratio']:,.2f}"
        data["commission_pnl_ratio"] = f"{data['commission_pnl_ratio']:,.2f}"
        data["pnl_drawdown_ratio"] = f"{data['pnl_drawdown_ratio']:,.2f}"
        data["pnl_std_ratio"] = f"{data['pnl_std_ratio']:,.2f}%"

        data["total_count"] = f"{data['total_count']}"
        data["profit_count"] = f"{data['profit_count']}"
        data["loss_count"] = f"{data['loss_count']}"
        data["average_trade_profit"] = f"{data['average_trade_profit']}"
        data["average_trade_loss"] = f"{data['average_trade_loss']}"

        for key, cell in self.cells.items():
            value = data.get(key, "")
            cell.setText(str(value))


class OptimizationResultMonitor(OriginalOptimizationResultMonitor):
    """"""

    def init_ui(self):
        """"""
        self.setWindowTitle("参数优化结果")
        self.resize(1100, 500)

        # Creat table to show result
        table = QtWidgets.QTableWidget()

        table.setColumnCount(15)
        table.setRowCount(len(self.result_values))
        horizontalHeader_list = ["参数",
                                 self.target_display,
                                 "总收益率",
                                 "年化收益",
                                 "最大回撤",
                                 "百分比最大回撤",
                                 "总成交笔数",
                                 "夏普比率",
                                 "胜率（交易日）",
                                 "盈亏比（交易日）",
                                 "交易结果次数",
                                 "盈利结果次数",
                                 "亏损结果次数",
                                 "平均盈利金额",
                                 "平均亏损金额"]
        table.setHorizontalHeaderLabels(horizontalHeader_list)
        table.setEditTriggers(table.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )

        for i in range(1, 15):
            table.horizontalHeader().setSectionResizeMode(
            i, QtWidgets.QHeaderView.Stretch
            )

        # print("--------------------修改后的init_ui--------------------")

        for n, tp in enumerate(self.result_values):
            setting, target_value, other_value = tp
            setting_cell = QtWidgets.QTableWidgetItem(str(setting))
            target_cell = QtWidgets.QTableWidgetItem(f"{target_value:.2f}")

            # 新增字段
            total_return_cell = QtWidgets.QTableWidgetItem(str(other_value["total_return"]))
            annual_return_cell = QtWidgets.QTableWidgetItem(str(other_value["annual_return"]))
            max_drawdown_cell = QtWidgets.QTableWidgetItem(str(other_value["max_drawdown"]))
            max_ddpercent_cell = QtWidgets.QTableWidgetItem(str(other_value["max_ddpercent"]))
            total_trade_count_cell = QtWidgets.QTableWidgetItem(str(other_value["total_trade_count"]))
            sharpe_ratio_cell = QtWidgets.QTableWidgetItem(str(other_value["sharpe_ratio"]))
            profit_days_ratio_cell = QtWidgets.QTableWidgetItem(str(other_value["profit_days_ratio"]))
            profit_loss_ratio_cell = QtWidgets.QTableWidgetItem(str(other_value["profit_loss_ratio"]))
            total_count_cell = QtWidgets.QTableWidgetItem(str(other_value["total_count"]))
            profit_count_cell = QtWidgets.QTableWidgetItem(str(other_value["profit_count"]))
            loss_count_cell = QtWidgets.QTableWidgetItem(str(other_value["loss_count"]))
            average_trade_profit_cell = QtWidgets.QTableWidgetItem(str(other_value["average_trade_profit"]))
            average_trade_loss_cell = QtWidgets.QTableWidgetItem(str(other_value["average_trade_loss"]))

            setting_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            target_cell.setTextAlignment(QtCore.Qt.AlignCenter)

            # 新增字段
            total_return_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            annual_return_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            max_drawdown_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            max_ddpercent_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            total_trade_count_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            sharpe_ratio_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            profit_days_ratio_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            profit_loss_ratio_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            total_count_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            profit_count_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            loss_count_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            average_trade_profit_cell.setTextAlignment(QtCore.Qt.AlignCenter)
            average_trade_loss_cell.setTextAlignment(QtCore.Qt.AlignCenter)

            table.setItem(n, 0, setting_cell)
            table.setItem(n, 1, target_cell)

            # 新增字段
            table.setItem(n, 2, total_return_cell)
            table.setItem(n, 3, annual_return_cell)
            table.setItem(n, 4, max_drawdown_cell)
            table.setItem(n, 5, max_ddpercent_cell)
            table.setItem(n, 6, total_trade_count_cell)
            table.setItem(n, 7, sharpe_ratio_cell)
            table.setItem(n, 8, profit_days_ratio_cell)
            table.setItem(n, 9, profit_loss_ratio_cell)
            table.setItem(n, 10, total_count_cell)
            table.setItem(n, 11, profit_count_cell)
            table.setItem(n, 12, loss_count_cell)
            table.setItem(n, 13, average_trade_profit_cell)
            table.setItem(n, 14, average_trade_loss_cell)

        # Create layout
        button = QtWidgets.QPushButton("保存")
        button.clicked.connect(self.save_csv)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(button)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(table)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def save_csv(self) -> None:
        """
        Save table data into a csv file
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV(*.csv)")

        if not path:
            return

        horizontalHeader_list = ["参数",
                                 self.target_display,
                                 "总收益率",
                                 "年化收益",
                                 "最大回撤",
                                 "百分比最大回撤",
                                 "总成交笔数",
                                 "夏普比率",
                                 "胜率（交易日）",
                                 "盈亏比（交易日）",
                                 "交易结果次数",
                                 "盈利结果次数",
                                 "亏损结果次数",
                                 "平均盈利金额",
                                 "平均亏损金额"]

        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator="\n")

            writer.writerow(horizontalHeader_list)

            for tp in self.result_values:
                setting, target_value, other_value = tp
                row_data = [str(setting),
                            str(target_value),
                            str(other_value["total_return"]),
                            str(other_value["annual_return"]),
                            str(other_value["max_drawdown"]),
                            str(other_value["max_ddpercent"]),
                            str(other_value["total_trade_count"]),
                            str(other_value["sharpe_ratio"]),
                            str(other_value["profit_days_ratio"]),
                            str(other_value["profit_loss_ratio"]),
                            str(other_value["total_count"]),
                            str(other_value["profit_count"]),
                            str(other_value["loss_count"]),
                            str(other_value["average_trade_profit"]),
                            str(other_value["average_trade_loss"])]
                writer.writerow(row_data)
