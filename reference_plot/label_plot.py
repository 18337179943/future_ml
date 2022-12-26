import sys, os
sys.path.insert(0, 'D:/策略开发/futures_ml/')
from datas_process.m_datas_process import *
from backtesting import BackTester
from backtesting.data_analyze_show import indicshow
import pandas as pd
__Author__ = 'ZCXY'


def train_datas_label_plot(symbol, pa, y_thread):
    
    _, suffix = run_dp(symbol, pa, y_thread=y_thread, index_n=10, method=0, need_svd=False)
    da_pa = f'{pa_prefix}/datas/data_set/{symbol}/normalize_datas_{suffix}'
    bt = BackTester()
    df = bt.signal_analyze(symbol, da_pa, save_pa=None, traindate=None, valdate=None, testdate=None, enddate=None, save_df=0, params='all')
    return df


if __name__ == '__main__':
    symbol = 'cf'
    pa = 'cf_60m_1.743_sample_20_1_return_rate_60m'
    y_thread = [7, 0.6, 1, 1]

    fig2 = plt.figure()
    ax2= fig2.add_axes([0.05, 0.75, 0.85, 0.2])
    ax5= fig2.add_axes([0.05, 0.5, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.2, 0.85, 0.2],sharex=ax2)
    # ax6= fig2.add_axes([0.05, 0.41, 0.85, 0.1],sharex=ax2)
    # ax7= fig2.add_axes([0.05, 0.28, 0.85, 0.1],sharex=ax2)
    # ax4= fig2.add_axes([0.05, 0.04, 0.85, 0.2], sharex=ax2)  ####left,bottom,width,height
    plt.title(f'{symbol}')
    data = train_datas_label_plot(symbol, pa, y_thread)
    print(data[data['cost']!=0].shape)
    zs = indicshow(fig2, ax2, ax3, ax5, data)
    data.to_csv('data.csv')
    plt.show()