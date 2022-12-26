import subprocess
from os.path import join
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import operator
from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

import time
import json
import numpy as np
import datetime
from mpl_finance import candlestick_ohlc
from  matplotlib.widgets import MultiCursor
from m_base import *
# from mpl_finance import candlestick_ochl

def getJSON(filename):
    fd = open(filename, 'r')
    text = fd.read()
    fd.close()
    returndata = json.loads(text)
    return returndata
def getCoefFromJson(config):
    coef = []
    tcList =[]
    for k,v in config.items():
        if 'Lead' not in k:
            for i,w in enumerate(v['weight']):
                coef.append([k+'_%d'%(i+1), w])
        if 'TickLoc' in k:
            tcList = v['para'][1:-1]
            tcList = dict((i+1, tcList[i]) for i in range(10))
    coef = pd.DataFrame(coef, columns = ['name', 'coef'])
    return coef, tcList

def indicshow(fig2,ax2,ax3,ax5,data, ax6=None, mod=0):
    # print(data1[['AskPrice1','BidPrice1','aggrvalue','qty','pos','floating_profit',"threshod",'indicator']])
    # data=data1[['AskPrice1','BidPrice1','aggrvalue','qty','pos','floating_profit',"threshod",'indicator']].astype(float)
    # data=data.dropna(axis=0)
    data = data.fillna(0)
    data['close'] = [float(x) for x in np.array(data['close'])]
    data['signal'] = [float(x) for x in np.array(data['signal'])]
    data['pred_sig'] = [float(x) for x in np.array(data['pred_sig'])]
    data['pos'] = [float(x) for x in np.array(data['pos'])]
    # data['pnl'] = [float(x) for x in np.array(data['pnl'])]
    data['pnl_cost'] = [float(x) for x in np.array(data['pnl_cost'])]
    ax2.cla()
    ax3.cla()
    # ax4.cla()
    ax5.cla()
    # ax7.cla()
    ax2.grid()
    ax3.grid()
    # ax4.grid()
    ax5.grid()
    if ax6 is not None:
        ax6.cla()
        ax6.grid()
    # ax7.grid()
    ax2.plot(range(len(data)), list(data['close']))
    # ax2.plot(range(len(data)), list(data['BidPrice1']),color='w')
    data = data.dropna(axis=0)

    if mod == 0:
        for j in range(len(data)):
            if data['signal'].iloc[j]>0:
                ax2.plot(j, data['close'].iloc[j], '*r')
            if data['signal'].iloc[j]<0:
                ax2.plot(j, data['close'].iloc[j], '^g')

        ax3.set_title('pos')
        # ax4.set_title('indic')
        ax5.set_title('pnl')
        # ax6.set_title('threshold')
        # ax7.set_title('indic')
        ax5.plot(range(len(data)), list(data['pnl_cost']))
        ax3.plot(range(len(data)), list(data['pos']))
        return MultiCursor(fig2.canvas, (ax2,ax3,ax5), color='r', lw=1)

    elif mod == 1:
        for j in range(len(data)):
            if data['signal_class'].iloc[j] == 'trend':
                if data['signal'].iloc[j]>0:
                    ax2.plot(j, data['close'].iloc[j], '*r')
                else:
                    ax2.plot(j, data['close'].iloc[j], '^r')
            
            elif data['signal_class'].iloc[j] == 'revers':
                if data['signal'].iloc[j]>0:
                    ax2.plot(j, data['close'].iloc[j], '*g')
                else:
                    ax2.plot(j, data['close'].iloc[j], '^g')

            elif data['signal_class'].iloc[j] == 'adjust':
                if data['signal'].iloc[j]>0:
                    ax2.plot(j, data['close'].iloc[j], '*b')
                else:
                    ax2.plot(j, data['close'].iloc[j], '^b')

            elif data['signal_class'].iloc[j] == 'other':
                if data['signal'].iloc[j]>0:
                    ax2.plot(j, data['close'].iloc[j], '*y')
                else:
                    ax2.plot(j, data['close'].iloc[j], '^y')
        ax3.set_title('pos')
        # ax4.set_title('indic')
        ax5.set_title('pnl')
        # ax6.set_title('threshold')
        # ax7.set_title('indic')
        ax5.plot(range(len(data)), list(data['pnl_cost']))
        ax3.plot(range(len(data)), list(data['pos']))
        return MultiCursor(fig2.canvas, (ax2,ax3,ax5), color='r', lw=1)

    elif mod == 2:  
        for j in range(len(data)):
            if data['signal'].iloc[j]>0:
                ax2.plot(j, data['close'].iloc[j], '*r')
            if data['signal'].iloc[j]<0:
                ax2.plot(j, data['close'].iloc[j], '^g')
        
        ax3.plot(range(len(data)), list(data['close1']))
        for j in range(len(data)):
            if data['signal1'].iloc[j]>0:
                ax3.plot(j, data['close1'].iloc[j], '*r')
            if data['signal1'].iloc[j]<0:
                ax3.plot(j, data['close1'].iloc[j], '^g')

        # ax4.set_title('indic')
        ax5.set_title('pnl')
        # ax6.set_title('threshold')
        # ax7.set_title('indic')
        ax5.plot(range(len(data)), list(data['pnl_cost']))
        # ax3.plot(range(len(data)), list(data['pos']))
        return MultiCursor(fig2.canvas, (ax2,ax3,ax5), color='r', lw=1)

    elif mod == 3:
        for j in range(len(data)):
            if data['pred_sig'].iloc[j]>0:
                ax2.plot(j, data['close'].iloc[j], '*r')
            if data['pred_sig'].iloc[j]<0:
                ax2.plot(j, data['close'].iloc[j], '^g')

        ax3.set_title('pos')
        # ax4.set_title('indic')
        ax5.set_title('pnl')
        # ax6.set_title('threshold')
        # ax7.set_title('indic')
        ax5.plot(range(len(data)), list(data['pnl_cost']))
        ax3.plot(range(len(data)), list(data['pos']))
        return MultiCursor(fig2.canvas, (ax2,ax3,ax5), color='r', lw=1)

    elif mod == 4:
        for j in range(len(data)):
            if data['signal'].iloc[j]>0:
                ax2.plot(j, data['close'].iloc[j], '*r')
            if data['signal'].iloc[j]<0:
                ax2.plot(j, data['close'].iloc[j], '^g')

        ax3.set_title('pos')
        # ax4.set_title('indic')
        ax5.set_title('pnl')
        # ax6.set_title('threshold')
        # ax7.set_title('indic')
        ax5.plot(range(len(data)), list(data['pnl_cost']))
        ax3.plot(range(len(data)), list(data['pos']))
        ax6.plot(range(len(data)), list(data['index_val']))
        return MultiCursor(fig2.canvas, (ax2,ax3,ax5, ax6), color='r', lw=1)

def indicshow_index(fig2, ax2, ax3, data, mod=0):
    '''将价格和指标画在一张图上'''
    data = data.fillna(0)
    data['close'] = [float(x) for x in np.array(data['close'])]
    # data['pnl'] = [float(x) for x in np.array(data['pnl'])]
    ax2.cla()
    ax2.grid()
    
    ax3.cla()
    ax3.grid()
    # ax2.plot(range(len(data)), list(data['close']))
    # ax2.plot(range(len(data)), list(data['BidPrice1']),color='w')
    data = data.dropna(axis=0)

    ohlc = data[['datetime', 'open', 'high', 'low', 'close']]
    ohlc['datetime'] = range(len(ohlc))
    candlestick_ohlc(ax2, ohlc.values.tolist(), width=.7, colorup='red', colordown='green')

    col_li = data.iloc[:, 7:].columns

    if mod == 0:
        ax3.set_title('index')
        for j in range(data.iloc[:, 7:].shape[1]):
            ax3.plot(range(len(data)), list(data.iloc[:, 7+j]), label=col_li[j])
        ax3.legend()
        return MultiCursor(fig2.canvas, (ax2,ax3), color='r', lw=1)

    elif mod == 1:
        for j in range(data.iloc[:, 7:].shape[1]):
            ax2.plot(range(len(data)), list(data.iloc[:, 7+j]))
            ax3.plot(range(len(data)), list(data.iloc[:, 7+j]), label=col_li[j])
        ax3.legend()
        return MultiCursor(fig2.canvas, (ax2, ax3), color='r', lw=1)


def plot_show():
    li = ['train', 'val', 'test']
    symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'AL', 'SF', 'JD', 'JM', 'OI', 'RB', 'V', 'P', 'sn'] # fg
    symbol = 'AL0'

    for i in li:
        plt.close()
        fig2 = plt.figure()
        ax2= fig2.add_axes([0.05, 0.75, 0.85, 0.2])
        ax5= fig2.add_axes([0.05, 0.5, 0.85, 0.2],sharex=ax2)
        ax3= fig2.add_axes([0.05, 0.2, 0.85, 0.2],sharex=ax2)
        # ax6= fig2.add_axes([0.05, 0.41, 0.85, 0.1],sharex=ax2)
        # ax7= fig2.add_axes([0.05, 0.28, 0.85, 0.1],sharex=ax2)
        # ax4= fig2.add_axes([0.05, 0.04, 0.85, 0.2], sharex=ax2)  ####left,bottom,width,height

        plt.title(f'{symbol}_{i}')
        # data = pd.read_csv(f'{pa_prefix}/filter_results/v/res7/y_pred_[5, 0.5, 1, 1]_v_60m_1.2_sample_20_1_return_rate_60m_{i}_analyze.csv')
        # li = [f'y_pred_{i}_m_60m_1.5_1.533_sample_20_1_return_rate_60m' for i in ['[10, 1, 1, 1]', '[5, 1, 1, 1]', '[10, 0.5, 1, 1]', '[5, 0.5, 1, 1]', '[7, 0.6, 1, 1]']]
        # data = pd.read_csv(f'{pa_prefix}/filter_results/{symbol}/res8/{li[4]}_{i}_analyze.csv')
        
        # li = [f'y_pred_{i}_pp_60m_1.3_15_1_return_rate_60m' for i in ['[10, 1, 1, 1]', '[5, 1, 1, 1]', '[10, 0.5, 1, 1]', '[5, 0.5, 1, 1]', '[7, 0.6, 1, 1]']]
        # data = pd.read_csv(f'{pa_prefix}/simulation/optuna_params/{symbol}/{li[-1]}_{i}_analyze.csv')
        # data = pd.read_csv(f'{pa_prefix}/datas/ml_result/symbol_result_adj/params/[7, 0.6, 1, 1]_sn_60m_1.6_sample_20_1_return_rate_60-m/y_pred_[7, 0.6, 1, 1]_sn_60m_1.6_sample_20_1_return_rate_60m_{i}_analyze.csv')
        pa = f'{pa_prefix}/simulation/optuna_params/{symbol}/'
        pa_li = os.listdir(pa)
        pa_i = list(filter(lambda x: 'train_analyze' in x, pa_li))[0]
        pa_i = pa_i.replace('train', i)
        pa_i = f'{pa}{pa_i}'
        data = pd.read_csv(pa_i)

        print(len(data))
        zs = indicshow(fig2, ax2, ax3, ax5, data, mod=0)
        plt.show()

def plot_show1(symbol, pa=None, save_pa=None, mod=0):
    # symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    # symbol = 'AP'

    plt.close()
    fig2 = plt.figure(figsize=(18, 12))
    ax2= fig2.add_axes([0.05, 0.75, 0.85, 0.2])
    ax5= fig2.add_axes([0.05, 0.5, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.2, 0.85, 0.2],sharex=ax2)
    # ax6= fig2.add_axes([0.05, 0.41, 0.85, 0.1],sharex=ax2)
    # ax7= fig2.add_axes([0.05, 0.28, 0.85, 0.1],sharex=ax2)
    # ax4= fig2.add_axes([0.05, 0.04, 0.85, 0.2], sharex=ax2)  ####left,bottom,width,height

    plt.title(f'{symbol}')
    # data = pd.read_csv(f'{pa_prefix}/filter_results/v/res7/y_pred_[5, 0.5, 1, 1]_v_60m_1.2_sample_20_1_return_rate_60m_{i}_analyze.csv')
    # li = [f'y_pred_{i}_m_60m_1.5_1.533_sample_20_1_return_rate_60m' for i in ['[10, 1, 1, 1]', '[5, 1, 1, 1]', '[10, 0.5, 1, 1]', '[5, 0.5, 1, 1]', '[7, 0.6, 1, 1]']]
    # data = pd.read_csv(f'{pa_prefix}/filter_results/{symbol}/res8/{li[4]}_{i}_analyze.csv')
    
    # li = [f'y_pred_{i}_pp_60m_1.3_15_1_return_rate_60m' for i in ['[10, 1, 1, 1]', '[5, 1, 1, 1]', '[10, 0.5, 1, 1]', '[5, 0.5, 1, 1]', '[7, 0.6, 1, 1]']]
    # data = pd.read_csv(f'{pa_prefix}/simulation/optuna_params/{symbol}/{li[-1]}_{i}_analyze.csv')
    # data = pd.read_csv(f'{pa_prefix}/datas/ml_result/symbol_result_adj/params/[7, 0.6, 1, 1]_sn_60m_1.6_sample_20_1_return_rate_60-m/y_pred_[7, 0.6, 1, 1]_sn_60m_1.6_sample_20_1_return_rate_60m_{i}_analyze.csv')
    # pa = f'{pa_prefix}/simulation/optuna_params/total_test_pos_adj/df_test_{symbol}_adj.csv'
    # pa = f'{pa_prefix}/simulation/optuna_params/total_test_raw/df_test_{symbol}.csv'
    if pa is None:
        pa = f'{pa_prefix}/simulation/optuna_params/total_val_raw/df_test_{symbol}.csv'
    data = pd.read_csv(pa)

    print(len(data))
    zs = indicshow(fig2, ax2, ax3, ax5, data, mod=mod)
    if save_pa is not None:
        plt.savefig(save_pa)
        plt.close()
    else:
        plt.show()

def plot_show_index_res(symbol, pa=None, save_pa=None, mod=4):
    # symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    # symbol = 'AP'

    plt.close()
    fig2 = plt.figure(figsize=(23, 14))
    ax2= fig2.add_axes([0.05, 0.75, 0.85, 0.2])
    ax5= fig2.add_axes([0.05, 0.525, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.3, 0.85, 0.2],sharex=ax2)
    ax6= fig2.add_axes([0.05, 0.05, 0.85, 0.2],sharex=ax2)
    plt.title(f'{symbol}')
    if pa is None:
        pa = f'{pa_prefix}/simulation/optuna_params/total_val_raw/df_test_{symbol}.csv'
    data = pd.read_csv(pa)

    print(len(data))
    zs = indicshow(fig2, ax2, ax3, ax5, data, ax6, mod=mod)
    if save_pa is not None:
        plt.savefig(save_pa)
        plt.close()
    else:
        plt.show()

def plot_show_index(symbol, data, save_pa=None, mod=0):
    # symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    # symbol = 'AP'

    plt.close()
    fig2 = plt.figure(figsize=(18, 12))
    ax2= fig2.add_axes([0.05, 0.5, 0.9, 0.45])
    # ax5= fig2.add_axes([0.05, 0.5, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.1, 0.9, 0.3],sharex=ax2)

    plt.title(f'{symbol}')
    zs = indicshow_index(fig2, ax2, ax3, data, mod=mod)
    if save_pa is not None:
        plt.savefig(save_pa)
        plt.close()
    else:
        plt.show()

def plot_pnl_all(pa=f'{pa_prefix}/simulation/optuna_params/total/total_test_analyze.csv', save_pa=f'{pa_prefix}/simulation/optuna_params/total/plot_pnl_all.csv'):
    df = pd.read_csv(pa)
    df.set_index('datetime', inplace=True)
    ax = df.plot()
    fig = ax.get_figure()
    fig.savefig(save_pa)
    plt.close()

def plot_pnl_seperate(df_res_all, save_pa=None):
    if save_pa is None:
        save_pa = makedir(f'{pa_prefix}/simulation/optuna_params/total/')
    for col in df_res_all.columns:
        plt.close()
        ax = pd.DataFrame(df_res_all[col]).plot(figsize=(18, 12))
        fig = ax.get_figure()
        fig.savefig(f'{save_pa}{col}.png')
    df_res_all.to_csv(f'{save_pa}df_res_all.csv')

if __name__ == '__main__':
    # fig1, ax1 = plt.subplots()
    # plot_show()
    # plot_pnl_seperate()
    symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    symbol = 'AP'
    # for symbol in symbol_li:
    # plot_show1(symbol)
    pa = f'{pa_prefix}/simulation/optuna_params/madifrsi/df_test_RB.csv'
    plot_show_index_res(symbol, pa=pa, save_pa=None, mod=4)
