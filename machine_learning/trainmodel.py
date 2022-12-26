import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
import catboost
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from machine_learning.m_vnpy import *
import random
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import json
import os
from backtesting import *
import shutil
from datas_process.m_futures_factors import SymbolsInfo
from m_base import read_json, makedir
from datas_process.m_datas_process import BaseDataProcess
import matplotlib.pyplot as plt
from backtesting.model_statistics import ConcatStatistics

__Author__ = 'ZCXY'

random.seed(1)
# import sys
# sys.path.append("..")

class BaseModel():
    def __init__(self) -> None:
        self.model_pa = None
        self.model = None
        self.data_set = None

    def save_model(self, model, model_name='model', model_path=None):
        '''保存模型'''
        if model_path == None:
            model_path = self.model_pa
        
        makedir(self.model_pa)

        joblib.dump(model, f'{model_path}/{model_name}.pkl')

    def load_model(self, model_name='model', model_path=None):
        '''读取模型'''
        if model_path == None:
            model_path = self.model_pa
        
        self.model = joblib.load(f'{model_path}/{model_name}.pkl')

    def calculate_roc_auc(self, x, y, model=None, binary=0):
        '''计算roc，auc'''
        def format_one(x):
            # if not all(list(x)):
            #     print(x)
            if np.sum(x) != 1:
                x.iloc[-1] = 1-np.sum(x.iloc[:-1])
            return x
        
        if model == None:
            model = self.model

        y = pd.DataFrame(y)
        # x = x.loc[~y.isin([np.nan]).values, :]
        # y = y.loc[~y.isin([np.nan]).values, :]
        y_pred = model.predict_proba(x)
            
        y_pred = pd.DataFrame(y_pred).apply(format_one, axis=1).values
        # not_one = [i for i, j in enumerate(y_sum) if j != 1]

        # print('均值：', np.mean(np.sum(y_pred, axis=1)))
        if binary:
            # print(y['y'])
            # print(np.shape(y['y'].values))
            # print(y_pred)
            # print(np.shape(y_pred))
            acc = roc_auc_score(y['y'].values, y_pred[:, 1])  # {'raise', 'ovr', 'ovo'}
        else:
            acc = roc_auc_score(y['y'].values, y_pred, multi_class='ovo')  # {'raise', 'ovr', 'ovo'}
        return acc

    def calculate_roc_auc_all(self, binary=0):
        '''计算训练和验证集的roc，auc'''
        acc_train = self.calculate_roc_auc(self.data_set['x_train'], self.data_set['y_train'], binary=binary)
        acc_val = self.calculate_roc_auc(self.data_set['x_valid'], self.data_set['y_valid'], binary=binary)
        if self.need_test_set:
            acc_test = self.calculate_roc_auc(self.data_set['x_test'], self.data_set['y_test'], binary=binary)
        else:
            acc_test = 0
        return acc_train, acc_val, acc_test

    def calculate_precision(self, x, y):
        y_pred = self.model.predict(x)
        # acc = precision_score(y, y_pred, average='micro')
        acc = accuracy_score(y, y_pred)
        return acc
    
    def calculate_precision_all(self):
        '''计算训练和验证集的准确率'''
        acc_train = self.calculate_precision(self.data_set['x_train'], self.data_set['y_train'])
        acc_val = self.calculate_precision(self.data_set['x_valid'], self.data_set['y_valid'])
        if self.need_test_set:
            acc_test = self.calculate_precision(self.data_set['x_test'], self.data_set['y_test'])
        else:
            acc_test = 0
        return acc_train, acc_val, acc_test
    
    def accuracy_adj(self, x, y):
        y_pred = pd.DataFrame(self.model.predict(x))
        y_pred.columns = ['y_pred']
        y_merge = pd.merge(y, y_pred, left_index=True, right_index=True)
        y_merge = y_merge[y_merge.iloc[:, 0]!=1]
        acc = accuracy_score(y_merge.iloc[:, 0], y_merge.iloc[:, 1])
        return acc

    def calculate_accuracy_adj_all(self):
        '''只计算1和-1的准确率'''
        acc_train = self.accuracy_adj(self.data_set['x_train'], self.data_set['y_train'])
        acc_val = self.accuracy_adj(self.data_set['x_valid'], self.data_set['y_valid'])
        acc_test = self.accuracy_adj(self.data_set['x_test'], self.data_set['y_test'])
        return acc_train, acc_val, acc_test

    def pnl(self, x, model=None, target_type='drawdown'):
        '''年化收益率/夏普比率'''
        if model==None:
            model = self.model
        y_pred = pd.DataFrame(model.predict(x))
        if y_pred.shape[1] == 1:
            y_pred.columns = ['y_pred']
            y_pred['datetime'] = x.index
        elif y_pred.shape[1] == 2:
            pass
        # bt = BackTester()
        bt = MyBackTester()     # 修改_calss
        cs = ConcatStatistics()
        df_res, drawdown = bt.all_contract_backtesting(self.symbol, y_pred['datetime'].iloc[0], y_pred['datetime'].iloc[-1], y_pred.copy(), target_type=target_type)
        holding_rate = cs.caculate_holding_rate(df0=df_res)
        annual_return = cs.caculate_annual_return(df_res)

        # _, annual_return = bt.select_bactesting(self.symbol, y_pred['datetime'].iloc[0], y_pred['datetime'].iloc[-1], y_pred, target_type=target_type)
        return drawdown, holding_rate, annual_return
    
    def holding_rate_score(self, hr_li):
        ''''''
        max_holding_rate, min_hodling_rate = np.max(hr_li), np.min(hr_li)
        return 0 if max_holding_rate < 0.8 and min_hodling_rate > 0.2 else -3

    def calculate_pnl_all(self, target_type='drawdown'):
        '''所有年化收益率/夏普比率/相对回撤周期'''
        acc_train, hr_train, ar_train = self.pnl(self.data_set['x_train'], target_type=target_type)
        acc_val, hr_val, ar_val = self.pnl(self.data_set['x_valid'], target_type=target_type)
        if self.need_test_set:
            acc_test, hr_test, ar_test = self.pnl(self.data_set['x_test'], target_type=target_type)
        else:
            acc_test = 0
        hr_score = self.holding_rate_score([hr_train, hr_val])  # 根据持仓周期给分数
        if (ar_val * ar_train > 0 and (ar_train / ar_val > 5 or ar_train / ar_val < 0.2)) or hr_score == -3: 
            acc_train, acc_val, acc_test = -3, -3, -3
        return acc_train, acc_val, acc_test
    
    def pnl_test(self, x, model=None):
        '''年化收益率/夏普比率'''
        if model==None:
            model = self.model
        y_pred = pd.DataFrame(model.predict(x))
        if y_pred.shape[1] == 1:
            y_pred.columns = ['y_pred']
            y_pred['datetime'] = x.index
        elif y_pred.shape[1] == 2:
            pass
        # bt = BackTester()
        bt = MyBackTester()     # 修改_calss
        df_res, drawdown = bt.all_contract_backtesting(self.symbol, y_pred['datetime'].iloc[0], y_pred['datetime'].iloc[-1], y_pred.copy())
        total_profit = bt.get_target(target_type='total_profit', df_res=df_res)
        max_ddpercent = bt.get_target(target_type='max_ddpercent', df_res=df_res) 
        return drawdown, total_profit, max_ddpercent

    def calculate_pnl_all_test(self):
        '''所有年化收益率/夏普比率/相对回撤周期'''
        drawdown_train, total_profit_train, max_ddpercent_train = self.pnl_test(self.data_set['x_train'])
        drawdown_val, total_profit_val, max_ddpercent_val = self.pnl_test(self.data_set['x_valid'])
        drawdown_test, total_profit_test, max_ddpercent_test = self.pnl_test(self.data_set['x_test'])
        score_res = [drawdown_train, total_profit_train, max_ddpercent_train,
                     drawdown_val, total_profit_val, max_ddpercent_val,
                     drawdown_test, total_profit_test, max_ddpercent_test]
        return score_res
        
    def save_predict(self, x=None, y=None, model=None, save_name='y_pred'):
        '''保存预测结果'''
        def df_predict(x, y):
            y_pred = pd.DataFrame(model.predict_proba(x))
            if y_pred.shape[1] == 2:
                y_pred.columns = ['decline', 'rise']
            else: 
                y_pred.columns = ['decline', 'zero', 'rise']
            y_pred.index = x.index
            y_pred['y_pred'] = model.predict(x)
            y_pred['y_real'] = y.values
            print(y_pred['y_real'].value_counts())
            # y_pred['symbol'] = x['symbol']
            return y_pred

        if model == None:
            model = self.model

        if x == None:
            y_pred = df_predict(self.datas_all['x'], self.datas_all['y'])
            # y_train_pred = df_predict(self.data_set['x_train'], self.data_set['y_train'])
            # y_val_pred = df_predict(self.data_set['x_valid'], self.data_set['y_valid'])
            # y_test_pred = df_predict(self.data_set['x_test'], self.data_set['y_test'])
            # y_pred = pd.concat([y_train_pred, y_val_pred, y_test_pred])
        else:
            y_pred = df_predict(x)
 
        save_pa = f'{pa_prefix}/datas/predict/{self.symbol}/'
        makedir(save_pa)
        pa_predict = f'{save_pa}{save_name}.csv'
        y_pred.to_csv(pa_predict)
            
        return y_pred, pa_predict

    def show_importance(self, model=None, file_name='', is_save=1):
        '''特征重要性'''
        if model == None:
            model = self.model
        df = pd.DataFrame({'columns': self.data_set['x_train'].columns, 'importance': model.feature_importances_}).sort_values(by='importance')
        if is_save:
            pa = f'{pa_prefix}/datas/feature_importances/{self.symbol}/'
            makedir(pa)
            df.to_csv(pa+file_name+'_feature_importance.csv')
        return df

    def statistic_hyperparameter(self, pa=None, save_pa=None, method=0):
        '''统计超参数区间范围'''
        if pa is None: 
            if method==0:
                pa = f'{pa_prefix}/datas/ml_result/model_1.0/params/'
                save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_1.0/hist_params/')
            else:
                pa = f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/raw/'
                save_pa = makedir(f'{pa_prefix}/datas/ml_result/model_1.0/hist_params_1.0/')
        sy_li = os.listdir(pa)
        res_li = []
        if method == 0:
            for sy_i in sy_li:
                sy_pa_li = os.listdir(f'{pa}{sy_i}')
                for pa_i in sy_pa_li:
                    pa_i_li = os.listdir(f'{pa}{sy_i}/{pa_i}')
                    json_pa = filter_str('.json', pa_i_li, is_list=0)
                    res_dic = read_json(f'{pa}{sy_i}/{pa_i}/{json_pa}')
                    res_dic.update({'symbol': sy_i})
                    res_li.append(res_dic.copy())
        else:
            for sy_i in sy_li:
                pa_i_li = os.listdir(f'{pa}{sy_i}/')
                json_pa = filter_str('.json', pa_i_li, is_list=0)
                res_dic = read_json(f'{pa}{sy_i}/{json_pa}')
                res_dic.update({'symbol': sy_i})
                res_li.append(res_dic.copy())
        df_res = pd.DataFrame(res_li)
        df_res.to_csv(f'{save_pa}df_res.csv', index=False)
        df_res.drop(columns=["class_weight", "objective", 'symbol'], inplace=True)
        [m_plot_one_hist(pd.DataFrame(df_res[col]), col, save_pa=f'{save_pa}') for col in df_res.columns]
        return df_res

    def get_model_params(self, suffix='', is_save=0, pa=None, save_pa=None, x_thread_li=[]):
        '''获取模型参数的json文件'''
        threadhold_li = [0.7, 0.9, 1.1, 1.3, 1.5]
        if pa is None: pa = f'{pa_prefix}/datas/ml_result/model_1.0/params/'
        li = suffix.split('_')

        y_thread, symbol, x_thread = li[0], li[1], li[3]
        
        li = os.listdir(f'{pa}{symbol}')

        thread = threadhold_li[x_thread_li.index(eval(x_thread))]  # thread1.0
        pa_i = list(filter(lambda x: (y_thread in x) and (str(thread) in x), li))[0]
        
        js_pa = filter_str('.json', os.listdir(f'{pa}{symbol}/{pa_i}'), is_list=0)
        load_pa = f'{pa}{symbol}/{pa_i}/{js_pa}'
        if is_save:
            if save_pa is None: save_pa = makedir(f'{pa_prefix}/machine_learning/optuna_params/{symbol}/')
            shutil.copy(load_pa, f'{save_pa}best_bp_{suffix}.json')
        return read_json(load_pa)
    
    def del_model_file(self):
        pa_dic = {'data_index': f'{pa_prefix}/datas/data_index/',
              'data_set': f'{pa_prefix}/datas/data_set/',
              'datas_columns': f'{pa_prefix}/datas/datas_columns/',
              'feature_importances': f'{pa_prefix}/datas/feature_importances/',
              'predict': f'{pa_prefix}/datas/predict/',
              'model': f'{pa_prefix}/machine_learning/model/',
              'optuna_params': f'{pa_prefix}/machine_learning/optuna_params/',
              'res': f'{pa_prefix}/machine_learning/res/',
              'backtest_res': f'{pa_prefix}/datas/backtest_res/'}
        for _, pa_i in pa_dic.items():
            del_folder_file(pa_i)
        
        os.rmdir(pa_dic['data_index'])
        shutil.copytree(f'{pa_prefix}/datas/data_index0/', pa_dic['data_index'])
    

class TrainClassification(BaseModel):
    def __init__(self, symbol=None, suffix='', model_name='lightgbm', need_test_set=1, zigzag='', mypa={}):

        self.need_test_set = need_test_set
        if len(mypa):
            self.pa, self.model_pa = mypa['pa'], mypa['model_pa']
            self.pa_train, self.pa_val, self.pa_test = mypa['pa_train'], mypa['pa_val'], mypa['pa_test']
            self.pa_all = mypa['pa_all']
        else:
            self.pa = f'{pa_prefix}/datas/data_set/'
            self.model_pa = f'{pa_prefix}/machine_learning/model/{symbol}'
            self.pa_train = f'{self.pa}{symbol}/train_datas_{zigzag}{suffix}.csv'
            self.pa_val = f'{self.pa}{symbol}/val_datas_{zigzag}{suffix}.csv'
            if need_test_set:
                self.pa_test = f'{self.pa}{symbol}/test_datas_{zigzag}{suffix}.csv'
            self.pa_all = f'{self.pa}{symbol}/normalize_datas_{suffix}.csv'
        symbolsinfo = SymbolsInfo()
        self.symbol = symbol
        self.symbols = symbolsinfo.df_symbols_all['symbol'].to_list()
        self.data_set, self.datas_all = self.get_datasets()
        self.model_name = model_name

    def get_datasets(self):
        '''获取训练集，验证集和测试集'''
        x_train, y_train, df_train = self.get_xy(self.pa_train)
        x_valid, y_valid, df_valid = self.get_xy(self.pa_val)
        x_all, y_all, df_all = self.get_xy(self.pa_all)
        data_set = {'x_train': x_train,
                   'y_train': y_train,
                   'x_valid': x_valid,
                   'y_valid': y_valid}

        if self.need_test_set:
            x_test, y_test, df_test = self.get_xy(self.pa_test)
            data_set.update({'x_test': x_test,
                             'y_test': y_test})
        
        datas_all = {
            'x': x_all,
            'y': y_all,
            'xy': df_all
        }
        return data_set, datas_all

    def get_xy(self, pa):
        '''获取训练数据''' 
        df = pd.read_csv(pa)
        df = df.set_index('datetime')
        df_o = df.copy()
        df_y = df.pop('y')
        df_x = df
        # datas = lgb.Dataset(df_x, label=df_y, reference=ref, params={'zero_as_missing': True})
        # score = datas.get_init_score()
        return df_x, df_y, df_o
    
    def set_parameters(self, parameters, model_name=None):
        '''设置模型参数'''
        if model_name == None:
            model_name = self.model_name
        
        if model_name == 'lightgbm':
            self.model = LGBMClassifier( 
                                # device_type = 'gpu', 
                                n_jobs = 1, 
                                **parameters)
        elif model_name == 'catboost':
            parameters.pop('class_weight')
            parameters.pop('num_class')
            parameters.pop('colsample_bytree')
            parameters.pop('reg_alpha')
            parameters.pop('min_split_gain')
            # parameters.pop('categorical_feature')
            # parameters.pop('categorical_feature')
            self.model = CatBoostClassifier(**parameters)

    def get_best_params(self, path='adjust_parameters', file_name='parameters_2022_01_20_10_57_06'):
        '''获取模型最优超参'''
        study = joblib.load(f'{path}/{file_name}.pickle')
        try:
            parameters = study.best_trial.params
        except:
            parameters = study
        print(parameters)
        return parameters

    def catboost_train(self, 
                data_set = None, 
                sample_weight = None,
                early_stopping_rounds = None):
        '''
        datas_set = {'x_train': x_train,
                   'y_train': y_train,
                   'x_valid': x_valid,
                   'y_valid': y_valid,
                   'x_test': x_test,
                   'y_test': y_test}
        '''
        if data_set == None:
            data_set = self.data_set

        parameters = {
            'X': data_set['x_train'],
            'y': data_set['y_train'],
            'verbose': 200
        }
        if 'x_valid' in data_set.keys():
            parameters.update({'eval_set': [(data_set['x_valid'], data_set['y_valid'])]})
        if sample_weight is not None:
            parameters.update({'sample_weight': sample_weight})
        if early_stopping_rounds is not None:
            parameters.update({'early_stopping_rounds': early_stopping_rounds})

        self.model.fit(**parameters)
        return self.model

    def lgbm_train(self, 
                data_set = None, 
                sample_weight = None,
                early_stopping_rounds = None,
                categorical_feature = None):
        '''
        datas_set = {'x_train': x_train,
                   'y_train': y_train,
                   'x_valid': x_valid,
                   'y_valid': y_valid,
                   'x_test': x_test,
                   'y_test': y_test}
        '''
        if data_set == None:
            data_set = self.data_set

        parameters = {
            'X': data_set['x_train'],
            'y': data_set['y_train'],
            'verbose': 200,
            'categorical_feature': categorical_feature
        }
        if 'x_valid' in data_set.keys():
            parameters.update({'eval_set': [(data_set['x_valid'], data_set['y_valid'])]})
        if categorical_feature is not None:
            parameters.update({'categorical_feature': categorical_feature})
        if sample_weight is not None:
            parameters.update({'sample_weight': sample_weight})
        if early_stopping_rounds is not None:
            parameters.update({'early_stopping_rounds': early_stopping_rounds})

        self.model.fit(**parameters)
        return self.model

    def run_predict_res(self):
        '''跑每个品种的分数'''
        def get_df(df, symbol):
            x = df[df['symbol']==symbol].copy()
            del x['symbol']
            y = x['y']
            del x['y']
            return x, y

        res = []
        for symbol in self.symbols:
            
            x_train, y_train = get_df(self.datas_all['train'], symbol)
            x_valid, y_valid = get_df(self.datas_all['valid'], symbol)
            x_test, y_test = get_df(self.datas_all['test'], symbol)
            
            self.datas_set = {'x_train': x_train,
                            'y_train': y_train,
                            'x_valid': x_valid,
                            'y_valid': y_valid,
                            'x_test': x_test,
                            'y_test': y_test}

            print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)
            acc_train, acc_val, acc_test = self.calculate_roc_auc_all()
            li = [symbol, acc_train, acc_val, acc_test]
            print(li)
            res.append(li)
        res = pd.DataFrame(res)
        res.columns = ['symbol', 'acc_train', 'acc_val', 'acc_test']
        res.to_csv(f'{pa_prefix}/datas/predict/res.csv')


class TrainClassificationCV(TrainClassification):
    def __init__(self, symbol=None, suffix=''):
        self.pa = f'{pa_prefix}/datas/data_set/'
        self.model_pa = f'{pa_prefix}/machine_learning/model/{symbol}'
        self.pa_train = f'{self.pa}{symbol}/train_datas_{suffix}.csv'
        self.pa_test = f'{self.pa}{symbol}/test_datas_{suffix}.csv'
        self.pa_all = f'{self.pa}{symbol}/normalize_datas_{suffix}.csv'
        self.n_train = 3
        symbolsinfo = SymbolsInfo()
        self.symbol = symbol
        self.symbols = symbolsinfo.df_symbols_all['symbol'].to_list()
        self.data_set, self.datas_all = self.get_datasets()

    def get_datasets(self):
        '''获取训练集，验证集和测试集'''
        _, _, train_datas = self.get_xy(self.pa_train)
        x_test, y_test, test_datas = self.get_xy(self.pa_test)
        x_all, y_all, df_all = self.get_xy(self.pa_all)
        data_set = {}
        bdp = BaseDataProcess()
        len_train = np.min(train_datas['y'].value_counts())
        split_n = len_train // 10 * 4
        
        for i in range(self.n_train):
            x_train = bdp.sample_datas(train_datas) # 获取均等分的数据集
            len_x_train_i = len(x_train)    # 获取数据集的长度
            shuffle_li = np.random.permutation(len_x_train_i)   # 打乱样本
            y_train = x_train.pop('y')  # x，y分开
            data_set_i = {'x_train': x_train.iloc[shuffle_li[:split_n]],
                   'y_train': y_train.iloc[shuffle_li[:split_n]],
                   'x_valid': x_train.iloc[shuffle_li[split_n:]],
                   'y_valid': y_train.iloc[shuffle_li[split_n:]],
                   'x_test': x_test,
                   'y_test': y_test}
            data_set.update({f'{i}': data_set_i})

        datas_all = {
            'x': x_all,
            'y': y_all,
            'xy': df_all
        }
        return data_set, datas_all
    
    def set_parameters(self, parameters):
        '''设置模型参数'''
        self.model_li = [LGBMClassifier(**parameters) for i in range(self.n_train)]

    def lgbm_train(self, 
                data_set = None, 
                sample_weight = None,
                early_stopping_rounds = None,
                categorical_feature = None):
        '''
        datas_set = {'x_train': x_train,
                   'y_train': y_train,
                   'x_valid': x_valid,
                   'y_valid': y_valid,
                   'x_test': x_test,
                   'y_test': y_test}
        '''
        if data_set == None:
            data_set = self.data_set

        params_li = []
        for i in range(self.n_train):
            str_i = str(i)
            data_set_i = data_set[str_i]
            parameters = {
                'X': data_set_i['x_train'],
                'y': data_set_i['y_train'],
                'verbose': 200,
                'categorical_feature': categorical_feature
            }
            if 'x_valid' in data_set_i.keys():
                parameters.update({'eval_set': [(data_set_i['x_valid'], data_set_i['y_valid'])]})
            if categorical_feature is not None:
                parameters.update({'categorical_feature': categorical_feature})
            if sample_weight is not None:
                parameters.update({'sample_weight': sample_weight})
            if early_stopping_rounds is not None:
                parameters.update({'early_stopping_rounds': early_stopping_rounds})
            params_li.append(parameters)
            self.model_li[i].fit(**parameters)

        # [model.fit(**parameters) for model, parameters in zip(self.model_li, params_li)]

        return self.model_li

    def calculate_roc_auc_all(self):
        '''计算训练和验证集的roc，auc'''
        acc_train_li, acc_val_li, acc_test_li = [], [], []
        for i in range(self.n_train):
            str_i = str(i)
            acc_train_li.append(self.calculate_roc_auc(self.data_set[str_i]['x_train'], self.data_set[str_i]['y_train'], self.model_li[i]))
            acc_val_li.append(self.calculate_roc_auc(self.data_set[str_i]['x_valid'], self.data_set[str_i]['y_valid'], self.model_li[i]))
            if self.need_test_set:
                acc_test_li.append(self.calculate_roc_auc(self.data_set[str_i]['x_test'], self.data_set[str_i]['y_test'], self.model_li[i]))
            else:
                acc_test_li.append(0)
            print('model', i, acc_train_li[-1], acc_val_li[-1], acc_test_li[-1])
        return acc_train_li, acc_val_li, acc_test_li


def run_train(symbol=None, parameters=None, suffix='', method=0, model_name='lightgbm', binary=0, need_test_set=1, target_type='drawdown'):
    '''训练和保存模型'''
    if parameters is None:
        parameters = read_json(f'{pa_prefix}/machine_learning/optuna_params/{symbol}/best_bp_{suffix}.json')
    # if symbol == None:
    #     parameters = read_json('{pa_prefix}/machine_learning/optuna_params/best_bp_'+symbol+'.json')
    #     file_name = 'all'
    # elif parameters == None:
    #     parameters = read_json('{pa_prefix}/machine_learning/optuna_params/best_bp.json')
    #     file_name = symbol
    # else:
    #     file_name = symbol
    if method:
        s = TrainClassificationCV(symbol=symbol, suffix=suffix, model_name=model_name) 
    else:
        s = TrainClassification(symbol=symbol, suffix=suffix, model_name=model_name, need_test_set=need_test_set)
    s.set_parameters(parameters)
    s.lgbm_train(categorical_feature=None, early_stopping_rounds=200)  # ['sector']
    acc_train, acc_val, acc_test = s.calculate_pnl_all(target_type=target_type)
    # acc_train, acc_val, acc_test = s.calculate_roc_auc_all(binary=binary)
    acc_train_p, acc_val_p, acc_test_p = s.calculate_precision_all()
    acc_all = s.calculate_roc_auc(s.datas_all['x'], s.datas_all['y'], binary=binary)
    # acc_train_adj, acc_train_adj, acc_train_adj = s.calculate_accuracy_adj_all()

    if binary == 1:
        acc_train_adj, acc_train_adj, acc_train_adj = 0, 0, 0           # 已改回来
    else:
        acc_train_adj, acc_train_adj, acc_train_adj = s.calculate_pnl_all()
    
    s.save_model(s.model, model_name='model_'+suffix)
    y_pred, pa_predict = s.save_predict(save_name='y_pred_'+suffix)
    df = s.show_importance(file_name=suffix)
    res_dic = {'acc_train': acc_train, 'acc_val': acc_val, 'acc_test': acc_test, 
               'acc_train_p': acc_train_p, 'acc_val_p': acc_val_p, 'acc_test_p': acc_test_p, 
               'acc_train_adj': acc_train_adj, 'acc_train_adj': acc_train_adj, 'acc_train_adj': acc_train_adj}
    print(df)
    print(y_pred)
    print(symbol, acc_train, acc_val, acc_test, acc_all)
    print('准确率：', acc_train_p, acc_val_p, acc_test_p)
    print('涨跌的准确率：', acc_train_p, acc_val_p, acc_test_p)
    return res_dic, pa_predict

def run_predict():
    '''模型跑分'''
    s = TrainClassification()
    s.load_model()
    s.run_predict_res()

def run():
    bm = BaseModel()
    bm.statistic_hyperparameter()

if __name__ == '__main__':

    '''训练和保存模型'''
    # run_train(suffix='sharp_ratio_30m_5m')

    '''保存预测结果'''
    # s = TrainClassification()
    # s.load_model()
    # y_pred = s.save_predict(save_name='y_pred1')
    # print(y_pred)

    # print(s.df_train.shape, s.df_val, s.df_test.shape)
    # run_predict()

    '''模型超参分布图'''
    bm = BaseModel()
    bm.statistic_hyperparameter(method=1)

