import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
sys.path.insert(0, pa_sys)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def run_gplearn_analyze():
    pa = f'{pa_prefix}/simulation/optuna_params/good_total_return_all_quantile_normal_2/remain/'
    file_li = os.listdir(pa)
    res_li = []
    for i in file_li:
        df = pd.read_csv(f'{pa}{i}/df_res_all.csv')
        re_i = df.iloc[-1].to_list()
        sy_i = re_i[1:-1]
        pnl_i = re_i[-1]
        num_max = len(list(filter(lambda x: x>1, sy_i)))
        num_min = len(list(filter(lambda x: x<1, sy_i)))
        res_max = np.max(sy_i)
        res_min = np.min(sy_i)
        res_dic = {'name': ['大于0个数', '小于0个数', '最大值', '最小值', '总收益'], 'value': [num_max, num_min, res_max, res_min, pnl_i]}
        pd.DataFrame(res_dic).to_csv(f'{pa}{i}/res_statistic.csv', index=False)
        df_res = pd.DataFrame({'val': sy_i})
        df_res.plot(figsize=(18, 12))
        plt.savefig(f'{pa}{i}/plot_res_statistic.png')
        plt.close()
        print(i)

    for i in file_li:
        df_i = pd.read_csv(f'{pa}{i}/res_statistic.csv')
        df_i.loc[-1] = ['指标名', i]
        df_i.set_index('name', inplace=True)
        df_i = df_i.T
        res_li.append(df_i)
        print(i)
    df_res = pd.concat(res_li, ignore_index=True).sort_values('总收益', ascending=False)
    df_res.to_csv(f'{pa}total_statistic.csv', index=False)


if __name__ == '__main__':
    run_gplearn_analyze()
