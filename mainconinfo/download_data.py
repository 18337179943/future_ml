from vnpy.trader.object import HistoryRequest
from vnpy.trader.utility import extract_vt_symbol
from vnpy.trader.constant import Interval
# from vnpy_rqdata.rqdata_datafeed import RqdataDatafeed
# from vnpy_sqlite.sqlite_database import SqliteDatabase
from datetime import datetime
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.database import get_database


datafeed = get_datafeed()
database = get_database()


def download_data(vt_symbol, interval, start, end):
    print("-" * 40)
    print('开始下载{}数据'.format(vt_symbol))
    symbol, exchange = extract_vt_symbol(vt_symbol)
    req = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        interval=Interval(interval),
        start=start,
        end=end
    )
    data = datafeed.query_bar_history(req)
    database.save_bar_data(data)
    print(vt_symbol, '数据保存完成')


if __name__ == '__main__':
    # vt_symbol = 'zn88.SHFE'
    interval = '1m'
    start = datetime(2021, 1, 1)
    end = datetime(2022, 1, 17)
    # symbols = ['al88.SHFE', 'j88.DCE', 'p88.DCE', 'cu88.SHFE', 'zn88.SHFE', 'ag88.SHFE', 'SF88.CZCE',
    #            'bu88.SHFE', 'ZC88.CZCE', 'MA88.CZCE', 'sc88.INE', 'IH88.CFFEX', 'IC88.CFFEX',
    #            'IF88.CFFEX', 'i88.DCE', 'ni88.SHFE', 'm88.DCE', 'au88.SHFE', 'sn88.SHFE', 'rb88.SHFE',
    #            'ru88.SHFE', 'FG88.CZCE', 'jm88.DCE']
    # symbols = ['j888.DCE', 'rb888.SHFE']

    '''43个常用品种'''
    symbols = ['a888.DCE', 'ag888.SHFE', 'al888.SHFE', 'AP888.CZCE', 'au888.SHFE', 'bu888.SHFE', 'c888.DCE', 'CF888.CZCE',
     'cs888.DCE', 'cu888.SHFE', 'eg888.DCE', 'FG888.CZCE', 'hc888.SHFE', 'i888.DCE', 'IC888.CFFEX', 'IF888.CFFEX',
     'IH888.CFFEX', 'j888.DCE', 'jd888.DCE', 'jm888.DCE', 'l888.DCE', 'm888.DCE', 'MA888.CZCE', 'ni888.SHFE',
     'OI888.CZCE', 'p888.DCE', 'pb888.SHFE', 'pg888.DCE', 'pp888.DCE', 'rb888.SHFE', 'RM888.CZCE', 'ru888.SHFE',
     'sc888.INE', 'SF888.CZCE', 'sn888.SHFE', 'SR888.CZCE', 'T888.CFFEX', 'TA888.CZCE', 'TF888.CFFEX',
     'v888.DCE', 'y888.DCE', 'ZC888.CZCE', 'zn888.SHFE']
    # symbols = ['IF888.CFFEX', 'IC888.CFFEX', 'IH888.CFFEX']
    for s in symbols:
        download_data(s, interval, start, end)
