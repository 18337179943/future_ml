import sys, os
from xmlrpc.client import TRANSPORT_ERROR
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from my_gplearn.genetic import SymbolicTransformer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import glob
from search_factor.my_gplearn.functions import function_set
import pandas as pd
__Author__ = 'ZCXY'
from m_base import makedir
# import gplearn as gp

# files = glob.glob('sectional_pred/*')
# for file in files:
#     os.remove(file)

is_normal = 3
pa = makedir(f'{pa_prefix}/search_factor/my_gplearn/raw_data/')
pa_good = makedir(f'{pa_prefix}/search_factor/my_gplearn/good_total_return_all_quantile_normal_{is_normal}_7030/')
save_pa = f'{pa_prefix}/search_factor/my_gplearn/res_data/'
# neutral_fac_path = 'data/neutral_fac20.mat'  #中性化因子,包括行业,市值,动量,波动率,换手率等
suffix = f'_normal_{is_normal}' if is_normal else ''


# 无处理k线
X = np.load(f'{pa}X{suffix}.npy')  #三维数据,第一维是交易日(对应的日期在data/date.xlsx),第二维是股票(对应的d股票代码在data/code.xlsx),第三维是个股的日频量价信息('open', 'high', 'low', 'close', 'vwap', 'volume', 'return1', 'free_turn', 'turn')
y = np.load(f'{pa}y{suffix}.npy')  #期货5小时后的收益率

# k线除以收盘价




print(X.shape, y.shape)

print('data loaded')

# def _logical(x1,x2,x3,x4):  # 自定义函数
#     return np.where(x1 > x2,x3,x4)
# logical = gp.functions.make_function(function = _logical,name = 'logical',arity = 4)

# function_set = ['neg', 'add', 'sub', 'mul', 'div','rank_add', 'rank_sub','decay_linear',
#                 'rank_mul', 'rank_div', 'ts_max', 'ts_min', 'ts_nanmean', 'ts_prod',
#                 'ts_rank', 'rank', 'ts_stddev', 'ts_sum', 'ts_corr', 'ts_cov','delta',
#                 'delay', 'sigmoid', 'sign', 'ts_skewness','ts_kurtosis','ts_max_diff',
#                 'ts_min_diff','ts_return','ts_zscore','ts_scale',
#                 'ts_min_max_cps','ts_ir','ts_median', 'winsorize', 'zscore', 'ts_argmax', 'ts_argmin']

# function_set = ['sigmoid', 'add']  # sign

gp = SymbolicTransformer(generations=100, population_size=2000,
                         hall_of_fame=200, n_components=100,
                         function_set=function_set,
                         parsimony_coefficient=0.0,
                         max_samples=1, verbose=1, 
                         random_state=6878,
                         init_depth=(1, 3),
                         metric = 'total_return_all_quantile',  # 'rank_ic',m_spearman total_return total_return_all_quantile rank_ic cumsum_return_rate
                         const_range = None,
                         tournament_size=20,
                         feature_names = ['open', 'high', 'low', 'close', 'volume'],
                         p_crossover = 0.4,
                         p_subtree_mutation=0.02,
                         p_hoist_mutation=0.02,
                         p_point_mutation=0.02,
                         p_point_replace=0.4,
                         save_thread=1.01,  # 1.01  -0.635
                         n_jobs=5)

gp.fit(X, y)

# for program in gp:
#     print(program)
#     print(program.raw_fitness_)
#     print(program.oob_fitness_)
#     print("--------------------------")

print('qqqqqqqqqqqqqqqqqq')
print(gp._best_programs)
df_res = pd.DataFrame()
for i, pr in enumerate(gp._best_programs):
    print(pr, str(pr), pr.raw_fitness_, pr.oob_fitness_, i)
    df_res = df_res.append({'index_name': str(pr), 'fitness': pr.raw_fitness_})
df_res.to_csv(f'{save_pa}total_return.csv', index=False)
print('done.')
