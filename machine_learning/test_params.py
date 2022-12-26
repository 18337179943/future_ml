import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *

from matplotlib.pyplot import step
import optuna
from optuna.pruners import ThresholdPruner
from optuna import TrialPruned
import joblib
from datetime import datetime
from machine_learning.trainmodel import TrainClassification, TrainClassificationCV, run_train
from m_base import makedir, save_json
from datas_process.m_datas_process import run_dp
import pandas as pd
import numpy as np
from backtesting.ctabacktester import BackTester
from machine_learning.trainmodel import BaseModel
import shutil
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
# from deep_learning.lstmmodel import *

__Author__ = 'ZCXY' 

class TestParams():
    '''测试每个模型参数对结果的敏感度'''
    def __init__(self, pa=f'{pa_prefix}/datas/ml_result/model_2.0/original_same_value_10_2/params/'):
        self.pa = pa
        self.save_pa = makedir(f'{pa_prefix}/datas/test_params_datas/')

    def get_symbol_params_result(self, symbol, params_range: dict):
        '''获取单个模型结果'''
        param_name = list(params_range.keys())[0]
        params_range_li = params_range[param_name]
        pa_li = os.listdir(f'{self.pa}{symbol}')
        df_concat = pd.DataFrame()
        for pa_i in pa_li:
            mp_pa_i = f'{self.pa}{symbol}/{pa_i}/'
            sy_pa_li = os.listdir(mp_pa_i)
            model_params_pa = filter_str('.json', sy_pa_li, is_list=0)
            model_params = read_json(f'{mp_pa_i}{model_params_pa}')
            if param_name == 'num_leaves': model_params.update({'max_depth': -1})
            train_pa = filter_str('train_datas_', sy_pa_li)
            val_pa = filter_str('val_datas_', sy_pa_li)
            test_pa = filter_str('test_datas_', sy_pa_li)
            normalize_pa = filter_str('normalize_datas_', sy_pa_li)
            res_li = []
            for pr in params_range_li:
                print(symbol, pa_i, param_name, pr)
                mp = deepcopy(model_params)
                mp.update({param_name: pr})
                mypa = {'pa': mp_pa_i,
                        'model_pa': '',
                        'pa_train': f'{mp_pa_i}{train_pa}',
                        'pa_val': f'{mp_pa_i}{val_pa}',
                        'pa_test': f'{mp_pa_i}{test_pa}',
                        'pa_all': f'{mp_pa_i}{normalize_pa}'}
                res_dic = self.run_train(symbol, mp, mypa)
                res_dic.update({'param_name': param_name, 'param_values': pr, 'model_pa': model_params_pa, 'symbol': symbol, 'params': mp})
                res_li.append(deepcopy(res_dic))
            df_res = pd.DataFrame(res_li)
            # for i in range(1, len(df_res)):
            #     df_res.iloc[i, :self.index_n] = df_res.iloc[i, :self.index_n] - df_res.iloc[0, :self.index_n]
            df_concat = pd.concat([df_concat, df_res])
        return df_concat

    def run_train(self, symbol, parameters, mypa, suffix='', model_name='lightgbm'):
        '''训练模型'''
        s = TrainClassification(symbol=symbol, suffix=suffix, model_name=model_name, mypa=mypa)
        s.set_parameters(parameters)
        s.lgbm_train(categorical_feature=None, early_stopping_rounds=200)  # ['sector']
        score_res = s.calculate_pnl_all_test()
        acc_train_p, acc_val_p, acc_test_p = s.calculate_precision_all()  
        auc_train, auc_val, auc_test = s.calculate_roc_auc_all()
        res_dic = {'drawdown_train': score_res[0], 'drawdown_val': score_res[3], 'drawdown_test': score_res[6],
                'total_profit_train': score_res[1], 'total_profit_val': score_res[4], 'total_profit_test': score_res[7], 
                'max_ddpercent_train': score_res[2], 'max_ddpercent_val': score_res[5], 'max_ddpercent_test': score_res[8],
                'acc_train_p': acc_train_p, 'acc_val_p': acc_val_p, 'acc_test_p': acc_test_p,
                'auc_train': auc_train, 'auc_val': auc_val, 'auc_test': auc_test}
        self.index_n = len(res_dic)
        return res_dic

    def main(self, params_range=None):
        '''parameters = {
            # 固定参数
            'learning_rate': [0.05, 0.1], d
            'class_weight': ['balanced'], d
            # 'class_weight': trial.suggest_categorical('class_weight', [{0:5,1:1,2:5}]),
            'n_estimators': [i for i in range(1000, 50000, 5000)],  11个

            # 优化算法参数
            'objective': ['multiclassova'], d
            'num_class': [3], d
            # 'tree_learner': trial.suggest_categorical('tree_learner', ['serial', 'feature', 'data']),
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),

            # 降低过拟合
            'max_depth': [i for i in range(1000, 50000, 5000)],
            'num_leaves': trial.suggest_int('num_leaves', 2,  
                        2**trial.suggest_int('max_depth', 3, 12, step=1)-1, step=1),

            'min_child_samples': [i for i in range(21, 81, 10)], d
            'max_bin': [15,31,63,127,255], d

            'subsample':  [i for i in range(0.7, 1, 0.1)], d
            'colsample_bytree':  [i for i in range(0.7, 1, 0.1)], d

            'reg_alpha': trial.suggest_loguniform('reg_alpha',1e-2,1e3),
            'reg_lambda': trial.suggest_loguniform('reg_lambda',1e-2,1e3), 

            'min_split_gain': [i for i in range(0, 1, 0.2)], d
            'random_seed': trial.suggest_categorical('random_seed', [88])
        }'''
        
        sy_li = os.listdir(self.pa)
        # params_range_li = [{'objective': ['multiclass', 'multiclassova']}, {'class_weight': ['balanced', {0:5,1:1,2:5}]}]
        # params_range_li = [{'max_bin': [15,31,63,127,255]}, {'min_split_gain': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}]
        # params_range = {'subsample': [0.7, 0.8, 0.9, 1]}
        # for params_range in params_range_li:
        df_concat = pd.DataFrame()
        for symbol in sy_li:
            df_res = self.get_symbol_params_result(symbol, params_range)
            df_concat = pd.concat([df_concat, df_res])
        df_concat.to_csv(f'{self.save_pa}{list(params_range.keys())[0]}.csv')
        # self.get_index_hist(df_concat)
        return 0

    def get_index_hist(self, df: pd.DataFrame):
        '''获取指标结果分布图'''
        symbol_li = df['symbol'].unique().tolist()
        param_values_li = df['param_values'].unique().tolist()
        param_name = df['param_name'].iloc[0]
        col_li = df.columns.to_list()
        for symbol in symbol_li:
            df_i = df[df['symbol']==symbol]
            save_folder_pa = makedir(f'{self.save_pa}{symbol}/{param_name}/')
            for val_j in param_values_li:
                df_j = df_i[df_i['param_values']==val_j]
                [m_plot_one_hist(pd.DataFrame(df_j.iloc[:, i]), f'{col_li[i]}_{val_j}_{df_j.iloc[:, i].mean()}', save_folder_pa) 
                for i in range(self.index_n)]
        save_total_pa = makedir(f'{self.save_pa}total/{param_name}/')
        for val_j in param_values_li:
            df_j = df[df['param_values']==val_j]
            [m_plot_one_hist(pd.DataFrame(df_j.iloc[:, i]), f'{col_li[i]}_{val_j}_{df_j.iloc[:, i].mean()}', save_total_pa) for i in range(self.index_n)]
        return 0
    
    def multipro_main(self, max_workers=3):
        # params_range_li = [{'min_child_samples': [i for i in range(21, 81, 10)]}, {'subsample': [0.7, 0.8, 0.9, 1]}, 
        #                    {'colsample_bytree':  [0.7, 0.8, 0.9, 1]}, {'objective': ['multiclass', 'multiclassova']}, 
        #                    {'class_weight': ['balanced', {0:2,1:1,2:2}]}, {'max_bin': [15,31,63,127,255]}, 
        #                    {'min_split_gain': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}, {'learning_rate': [0.02, 0.05]}]
        params_range_li = [{'n_estimators': [i for i in range(1000, 50000, 5000)]}, {'num_leaves': [7, 15, 31, 48, 63, 90, 127,
                            200, 255, 350, 511, 800, 1023, 1500, 2047]}, {'reg_alpha': [0.0, 0.2, 0.4, 0.6, 0.8, 1, 5, 10, 30, 100, 200, 500]},
                            {'reg_lambda': [0.0, 0.2, 0.4, 0.6, 0.8, 1, 5, 10, 30, 100, 200, 500]}]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # max_workers=10
            executor.map(self.main, params_range_li)

    def plot_reg_target(self):
        '''画正则项和target的关系曲线'''
        save_pa = makedir(f'{self.save_pa}plot_reg/')
        df = pd.read_csv(f'{self.save_pa}class_weight_balanced.csv')
        df_res = df.iloc[:, :14]
        df_res.drop(columns=['total_profit_train', 'total_profit_val', 'total_profit_test'], inplace=True)
        col_li = df_res.columns.to_list()
        df_res['reg_alpha'] = df['params'].apply(lambda x: eval(x)['reg_alpha'])
        df_res['reg_lambda'] = df['params'].apply(lambda x: eval(x)['reg_lambda'])
        for i in range(len(col_li)):
            m_plot_corr(df_res[[col_li[i], 'reg_alpha']], f'{col_li[i]}_reg_alpha', save_pa)
            m_plot_corr(df_res[[col_li[i], 'reg_lambda']], f'{col_li[i]}_reg_lambda', save_pa)

        return df_res
    

    
def run_TestParams():
    tp = TestParams()
    # tp.main()
    tp.multipro_main(4)
    # tp.plot_reg_target()


if __name__ == '__main__':
    run_TestParams()