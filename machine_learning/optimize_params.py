import sys, os
from m_base import *
sys_name = 'windows'
pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)

import optuna
import joblib
from datetime import datetime
from trainmodel import TrainClassification, TrainClassificationCV, run_train
from m_base import makedir, save_json
from datas_process.m_datas_process import run_dp
import pandas as pd
__Author__ = 'ZCXY' 
import numpy as np


class OptimizeParams():
    def __init__(self, symbol=None, suffix='', method=0, model_name='lightgbm', need_test_set=1) -> None:
        self.suffix = suffix
        self.save_params = f'{pa_prefix}/machine_learning/optuna_params/{symbol}/'
        makedir(self.save_params)
        self.symbol = symbol
        self.method = method
        self.need_test_set = need_test_set
        self.model_name = model_name
        if method:
            self.classifier = TrainClassificationCV(symbol=symbol, suffix=suffix, model_name=model_name) 
        else:
            self.classifier = TrainClassification(symbol=symbol, suffix=suffix, model_name=model_name, need_test_set=need_test_set)
        # self.classifier = TrainClassificationCV(symbol=symbol, suffix=suffix) if method else TrainClassification(symbol=symbol, suffix=suffix)
        self.res_dic = {'symbol': [], 'best_score': [], 'best_params': []}

    def optimize_res(self):
        res1 = {'learning_rate': 0.1, 'class_weight': 'balanced', 'n_estimators': 26000, 'objective': 'multiclass', 'num_class': 3, 'max_depth': 3, 'num_leaves': 2, 'min_child_samples': 21, 'max_bin': 15, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.30851202722659554, 'reg_lambda': 0.02566212324211507, 'min_split_gain': 0.0, 'random_seed': 88, 'target_n': 1.0}
        res2 = {'learning_rate': 0.1, 'class_weight': 'balanced', 'n_estimators': 41000, 'objective': 'multiclass', 'num_class': 3, 'max_depth': 3, 'num_leaves': 2, 'min_child_samples': 51, 'max_bin': 31, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 779.2298132690286, 'reg_lambda': 0.1397669308908931, 'min_split_gain': 1.0, 'random_seed': 88, 'target_n': 1.0}
        res3 = {'learning_rate': 0.1, 'class_weight': 'balanced', 'n_estimators': 21000, 'objective': 'multiclassova', 'num_class': 3, 'max_depth': 3, 'num_leaves': 2, 'min_child_samples': 31, 'max_bin': 63, 'subsample': 0.8999999999999999, 'colsample_bytree': 0.7999999999999999, 'reg_alpha': 576.3522297261052, 'reg_lambda': 140.8788487411502, 'min_split_gain': 0.8, 'random_seed': 88, 'target_n': 1.0}
        return res1

    def target_func(self, trial, study, train_data_set=None):
        '''????????????'''
        parameters = {
            # ????????????
            # 'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, step=0.02),

            'learning_rate': trial.suggest_categorical('learning_rate', [0.1]),
            # 'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
            # 'class_weight': trial.suggest_categorical('class_weight', [{0:5,1:1,2:5}]),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 50000, step=5000),

            # ??????????????????
            'objective': trial.suggest_categorical('objective', ['multiclass', 'multiclassova']),
            'num_class': trial.suggest_categorical('num_class', [3]),
            # 'tree_learner': trial.suggest_categorical('tree_learner', ['serial', 'feature', 'data']),
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),

            # ???????????????
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
        acc_train, acc_val, acc_test = self.classifier.calculate_pnl_all()
        # acc_train, acc_val, acc_test = self.classifier.calculate_precision_all()
        # target = acc_val
        # target = acc_val - trial.suggest_float('target_n', 1, 4, step=0.5)*abs(acc_train - acc_val)
        # target = val_score + 1.2*abs(train_score - val_score)

        target = self.target(acc_train, acc_val, trial)

        print(f"?????????: {acc_train} -- ?????????: {acc_val} -- ?????????: {acc_test}")
        if self.method:
            print(f'????????????????????????: {self.classifier.model_li[-1].best_iteration_} target: {target}')
        else:
            print(f'????????????????????????: {self.classifier.model.best_iteration_} target: {target}')

        return target

    def target(self, acc_train, acc_val, trial):
        '''???????????????'''
        if self.method == 0:
            target = acc_val - trial.suggest_float('target_n', 0.5, 1, step=0.5)*abs(acc_train - acc_val)
        elif self.method == 1:
            target = np.mean(acc_train) - trial.suggest_float('target_n', 1, 4, step=0.5)* \
                np.mean([abs(i-j) for i, j in zip(acc_train, acc_val)])
        elif self.method == 2:
            print('?????????????????????')
        return target

    def optuna_optimize(self, n_trials):
        '''optuna??????'''
        # try:
        target_direction='maximize'
        study = optuna.create_study(direction=target_direction)
        study.optimize(
            lambda trial : self.target_func(trial, study), n_trials=n_trials)

        bp = study.best_params
        best_score = study.best_trial.value

        # print('Number of finished trials:', len(study.trials))
        # print("------------------------------------------------")
        # pa_name = 'all' if self.symbol == None else self.symbol
        save_json(bp, f'{self.save_params}best_bp_{self.suffix}.json')

        print('Best trial: score {},\nparams {}'.format(best_score,bp))
        print("------------------------------------------------")
        acc_res = run_train(symbol=self.symbol, parameters=None, suffix=self.suffix, method=self.method, need_test_set=self.need_test_set)
        self.res_dic['symbol'].append(self.symbol)
        self.res_dic['best_score'].append(best_score)
        self.res_dic['best_params'].append(bp)
        self.res_dic.update(acc_res)
        del study

        return self.res_dic

    def optuna_optimize_one(self, syl):
        '''??????????????????'''
        pass

def run_optuna(symbol, suffix, y_thread, index_n, method, n_trials, model_name, need_svd=False, need_test_set=1):
    index_n = run_dp(symbol, suffix, y_thread, index_n, method, need_test_set, need_svd)
    s = OptimizeParams(symbol, f'{y_thread}_{suffix}', method, model_name, need_test_set)
    res_dic = s.optuna_optimize(n_trials)
    return res_dic, index_n

def main():
    # m_5m_0.6_50_1_return_rate_30m
    # ru_5m_0.6_50_1_sharp_ratio_60m
    symbol = 'v'
    method = 0  # target??????
    win_n = 60   # k?????????
    n_trials = 300    # ??????????????????
    del_interval = '_30m'
    save_interval = '_60m'  # y?????????
    model_name = 'lightgbm'   # 'lightgbm' 'catboost' ???????????????
    index_n = 5   # ?????????????????? 25
    need_svd = True
    need_test_set = 0
    y_thread = [7, 0.6, 1, 1]  # y????????? [5, 0.5, 1, 0] [10, 1, 1, 0] 0.7 [?????????????????????????????????, y??????0???1]
    suffix_li = os.listdir(f'{pa_prefix}/datas/data_index/{symbol}/')
    suffix_li = [i[:-4] for i in suffix_li]
    remove_li = [f'{symbol}_5m', f'{symbol}_15m', f'{symbol}_60m']
    for i in remove_li:
        if i in suffix_li:
            suffix_li.remove(i)
    # [suffix_li.remove(i) for i in remove_li]
    res_li = []
    dic_name = ['y_thread', 'symbol', 'interval', 'hist_threadhold', 'del', 'stationary', 'y']
    print(len(suffix_li))
    # for suffix in suffix_li:
    suffix = f'v_60m_1.2_sample_20_1_return_rate_60m'
    # suffix = f'{symbol}_{win_n}m_0.05_50_1_return_rate{save_interval}'
    print(suffix)
    res_dic, index_n = run_optuna(symbol, suffix, y_thread, index_n, method, n_trials, model_name, need_svd, need_test_set=need_test_set)
    suffix_i = f'{y_thread}_{suffix}'
    params = suffix_i.split('_', len(dic_name)-1)
    [res_dic.update({i:j}) for i, j in zip(dic_name, params)]
    res_dic.update({'index_n': index_n})
    res_li.append(res_dic)
    df_res = pd.concat([pd.DataFrame(i) for i in res_li])
    df_res.pop('del')
    pa = f'{pa_prefix}/machine_learning/res/'

    makedir(pa)
    df_res.to_csv(f'{pa}{symbol}{save_interval}_{y_thread}_{suffix}.csv', index=False)

if __name__ == '__main__':
    # symbol, suffix = 'M', 'm_5m_0.3_50_1_sharp_ratio_30m_and'
    # symbol, suffix = 'M', 'm_5m_0.6_50_0_mean_return_rate_30m'
    # symbol, suffix = 'M', 'm_5m_0.6_50_0_mean_return_rate_60m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_mean_return_rate_60m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_mean_return_rate_30m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_return_rate_30m'
    symbol, suffix = 'M', 'm_5m_0.6_50_1_return_rate_60m'
    symbol = 'M'
    main()
    