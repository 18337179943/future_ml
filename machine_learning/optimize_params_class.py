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
from backtesting.ctabacktester import BackTester, MyBackTester
import shutil
# from deep_learning.lstmmodel import *

__Author__ = 'ZCXY' 


class OptimizeParams():
    def __init__(self, symbol=None, suffix='', method=0, model_name='lightgbm', target_type='drawdown') -> None:
        self.suffix = suffix
        self.save_params = f'{pa_prefix}/machine_learning/optuna_params/{symbol}/'
        makedir(self.save_params)
        self.symbol = symbol
        self.method = method
        self.model_name = model_name
        self.target_type = target_type
        if method:
            self.classifier = TrainClassificationCV(symbol=symbol, suffix=suffix, model_name=model_name) 
        else:
            self.classifier = TrainClassification(symbol=symbol, suffix=suffix, model_name=model_name)
        # self.classifier = TrainClassificationCV(symbol=symbol, suffix=suffix) if method else TrainClassification(symbol=symbol, suffix=suffix)
        self.res_dic = {'symbol': [], 'best_score': [], 'best_params': []}

    def optimize_res(self):
        res1 = {'learning_rate': 0.1, 'class_weight': 'balanced', 'n_estimators': 26000, 'objective': 'multiclass', 'num_class': 3, 'max_depth': 3, 'num_leaves': 2, 'min_child_samples': 21, 'max_bin': 15, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.30851202722659554, 'reg_lambda': 0.02566212324211507, 'min_split_gain': 0.0, 'random_seed': 88, 'target_n': 1.0}
        res2 = {'learning_rate': 0.1, 'class_weight': 'balanced', 'n_estimators': 41000, 'objective': 'multiclass', 'num_class': 3, 'max_depth': 3, 'num_leaves': 2, 'min_child_samples': 51, 'max_bin': 31, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 779.2298132690286, 'reg_lambda': 0.1397669308908931, 'min_split_gain': 1.0, 'random_seed': 88, 'target_n': 1.0}
        res3 = {'learning_rate': 0.1, 'class_weight': 'balanced', 'n_estimators': 21000, 'objective': 'multiclassova', 'num_class': 3, 'max_depth': 3, 'num_leaves': 2, 'min_child_samples': 31, 'max_bin': 63, 'subsample': 0.8999999999999999, 'colsample_bytree': 0.7999999999999999, 'reg_alpha': 576.3522297261052, 'reg_lambda': 140.8788487411502, 'min_split_gain': 0.8, 'random_seed': 88, 'target_n': 1.0}
        return res1

    def target_func(self, trial, study, train_data_set=None, prune=0):
        '''目标函数'''
        parameters = {
            # 固定参数
            # 'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, step=0.02),

            'learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.1]),
            # 'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
            # 'class_weight': trial.suggest_categorical('class_weight', [{0:5,1:1,2:5}]),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 50000, step=5000),

            # 优化算法参数
            'objective': trial.suggest_categorical('objective', ['multiclass', 'multiclassova']),
            'num_class': trial.suggest_categorical('num_class', [3]),
            # 'tree_learner': trial.suggest_categorical('tree_learner', ['serial', 'feature', 'data']),
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),

            # 降低过拟合
            'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),
            'num_leaves': trial.suggest_int('num_leaves', 2,  
                        2**trial.suggest_int('max_depth', 3, 12, step=1)-1, step=1),

            'min_child_samples': trial.suggest_int('min_child_samples', 21,  81, step=10),
            'max_bin': trial.suggest_categorical('max_bin', [15,31,63,127,255]),

            'subsample':  trial.suggest_float('subsample',  0.7, 1, step=0.1),
            'colsample_bytree':  trial.suggest_float('colsample_bytree',  0.7, 1, step=0.1),

            'reg_alpha': trial.suggest_loguniform('reg_alpha',1e-2,1e3),
            'reg_lambda': trial.suggest_loguniform('reg_lambda',1e-2,1e3),

            'min_split_gain': trial.suggest_float('min_split_gain',  0, 1, step=0.2),
            'random_seed': trial.suggest_categorical('random_seed', [88])
        }
        self.classifier.set_parameters(parameters)

        if self.model_name == 'lightgbm':
            self.classifier.lgbm_train(train_data_set, 
                                sample_weight = None,
                                early_stopping_rounds = 200,
                                categorical_feature = None)
                                # categorical_feature = [trial.suggest_categorical('sector', ['sector', None])])
        else:
            self.classifier.catboost_train(train_data_set, 
                                sample_weight = None,
                                early_stopping_rounds = 200)
                                
        # acc_train, acc_val, acc_test = self.classifier.calculate_roc_auc_all()
        acc_train, acc_val, acc_test = self.classifier.calculate_pnl_all(target_type=self.target_type)
        
        # acc_train, acc_val, acc_test = self.classifier.calculate_precision_all()
        # target = acc_val
        # target = acc_val - trial.suggest_float('target_n', 1, 4, step=0.5)*abs(acc_train - acc_val)
        # target = val_score + 1.2*abs(train_score - val_score)

        target = self.target(acc_train, acc_val, trial)
        if str(target) == 'nan':
            target = -10

        print(f"品种：{self.symbol}, 训练集: {acc_train} -- 验证集: {acc_val} -- 测试集: {acc_test}")
        if self.method:
            print(f'模型最佳迭代次数: {self.classifier.model_li[-1].best_iteration_} target: {target}')
        else:
            print(f'模型最佳迭代次数: {self.classifier.model.best_iteration_} target: {target}')
        
        # if acc_train == -3 or acc_val == -3 or acc_test == -3:
        #     input()
        if prune:
            trial.report(target, 0)     # early stop
            if trial.should_prune(): raise TrialPruned()

        return target

    def target(self, acc_train, acc_val, trial):
        '''计算目标值'''
        if self.method == 0:
            target = acc_val - trial.suggest_float('target_n', 0.5, 1, step=0.5)*abs(acc_train - acc_val)
        elif self.method == 1:
            target = np.mean(acc_train) - trial.suggest_float('target_n', 1, 4, step=0.5)* \
                np.mean([abs(i-j) for i, j in zip(acc_train, acc_val)])
        elif self.method == 2:
            print('可以继续添加。')
        return target

    def optuna_optimize(self, n_trials, prune=0):
        '''optuna调参'''
        # try:
        target_direction='maximize'
        # study = optuna.create_study(direction=target_direction)
        if prune:
            study = optuna.create_study(direction=target_direction, pruner=ThresholdPruner(upper=-0.23))    # early stop
        else:
            study = optuna.create_study(direction=target_direction)    # early stop
        study.optimize(
            lambda trial : self.target_func(trial, study, prune=prune), n_trials=n_trials)

        bp = study.best_params
        best_score = study.best_trial.value

        # print('Number of finished trials:', len(study.trials))
        # print("------------------------------------------------")
        # pa_name = 'all' if self.symbol == None else self.symbol
        save_json(bp, f'{self.save_params}best_bp_{self.suffix}.json')

        print('Best trial: score {},\nparams {}'.format(best_score,bp))
        print("------------------------------------------------")
        acc_res, pa_predict = run_train(symbol=self.symbol, parameters=None, suffix=self.suffix, method=self.method, target_type=self.target_type)
        self.res_dic['symbol'].append(self.symbol)
        self.res_dic['best_score'].append(best_score)
        self.res_dic['best_params'].append(bp)
        self.res_dic.update(acc_res)
        del study

        return self.res_dic, pa_predict

    def optuna_optimize_one(self, syl):
        '''单个品种调参'''
        pass

def run_optuna(symbol, suffix, y_thread, index_n, method, n_trials, model_name, prune=0, target_type='drawdown'):
    '''单品种跑优化'''
    index_n, _ = run_dp(symbol, suffix, y_thread, index_n, method)
    s = OptimizeParams(symbol, f'{y_thread}_{suffix}', method, model_name, target_type=target_type)
    res_dic, pa_predict = s.optuna_optimize(n_trials, prune)
    return res_dic, index_n, pa_predict

def save_result(symbol, y_thread, suffix: str):
    save_dir = f'{pa_prefix}/datas/ml_result/{symbol}/{suffix}/'
    makedir(save_dir)
    pa_dic = {'data_index': f'{pa_prefix}/datas/data_index/{symbol}/',
              'data_set': f'{pa_prefix}/datas/data_set/{symbol}/',
              'datas_columns': f'{pa_prefix}/datas/datas_columns/{symbol}/',
              'feature_importances': f'{pa_prefix}/datas/feature_importances/{symbol}/',
              'predict': f'{pa_prefix}/datas/predict/{symbol}/',
              'model': f'{pa_prefix}/machine_learning/model/{symbol}/',
              'optuna_params': f'{pa_prefix}/machine_learning/optuna_params/{symbol}/',
              'res': f'{pa_prefix}/machine_learning/res/{symbol}/',
              'backtest_res': f'{pa_prefix}/datas/backtest_res/{symbol}/'}
    for dic_name, pa_index in pa_dic.items():
        if dic_name == 'data_index':
            suffix_i = suffix.replace(suffix.split('_')[0]+'_', '')
        else:
            suffix_i = suffix
        li = os.listdir(pa_index)
        pa_i = list(filter(lambda x: suffix_i in x, li))
        [print(f'{pa_index}{i}') for i in pa_i]
        [shutil.copy(f'{pa_index}{i}', save_dir) for i in pa_i]
    print('save_result:', save_dir, 'is done.')

def run_dp_optimize_all(symbol, threadhold_li, y_thread_li, n_trials=300):
    '''全品种跑优化'''
    method = 0  # target方法
    # n_trials = 300    # 优化迭代步数
    save_interval = '_60m'  # y的周期
    prune = 0   # optuna eraly stop
    # is_class = 0
    model_name = 'lightgbm'   # 'lightgbm' 'catboost' 还没有完成
    target_type = 'max_ddpercent'  # 'drawdown' max_ddpercent_duration
    index_n = 20   # 指标数量阈值 25
    res_li = []
    dic_name = ['y_thread', 'symbol', 'interval', 'hist_threadhold', 'del', 'stationary', 'y']
    pa_index_li = os.listdir(f'{pa_prefix}/datas/data_index/{symbol}/')
    pa_res = f'{pa_prefix}/machine_learning/res/{symbol}/'
    makedir(pa_res)
    bt = BackTester()
    # mbt = MyBackTester()  # 修改_class

    for y_thread in y_thread_li:
        for threadhold in threadhold_li:
            try:
                suffix = list(filter(lambda x: f'{threadhold}' in x, pa_index_li))[0][:-4]
            except:
                print(symbol, y_thread, threadhold, 'got wrong.')
                continue
            if suffix.split('_')[-5] == '0':
                print(suffix, 'got zero index.')
                continue
            # suffix = f'{symbol}{save_interval}_{threadhold}_sample_20_1_return_rate_60m'
            print(suffix)
            res_dic, index_n, pa_predict = run_optuna(symbol, suffix, y_thread, index_n, method, n_trials, model_name, prune=prune, target_type=target_type)  # 跑优化
            suffix_i = f'{y_thread}_{suffix}'
            params = suffix_i.split('_', len(dic_name)-1)
            [res_dic.update({i:j}) for i, j in zip(dic_name, params)]
            res_dic.update({'index_n': index_n})
            res_li.append(res_dic)  
            df_res = pd.concat([pd.DataFrame(i) for i in res_li])
            df_res.pop('del')
            df_res.to_csv(f'{pa_res}{y_thread}_{suffix}_res.csv', index=False)  # 保存优化参数
            try:
                bt.signal_analyze(symbol, pa_predict) # 跑模型回测
                # mbt.signal_analyze(symbol, pa_predict)  # 修改_class
            except:
                continue
            print('pa_predict', pa_predict)
            save_result(symbol, y_thread, f'{y_thread}_{suffix}')   # 保存模型相关文件
            print(suffix, 'is done.')


if __name__ == '__main__':
    # symbol, suffix = 'M', 'm_5m_0.3_50_1_sharp_ratio_30m_and'
    # symbol, suffix = 'M', 'm_5m_0.6_50_0_mean_return_rate_30m'
    # symbol, suffix = 'M', 'm_5m_0.6_50_0_mean_return_rate_60m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_mean_return_rate_60m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_mean_return_rate_30m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_return_rate_30m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_return_rate_60m'
    symbol = 'M'
    