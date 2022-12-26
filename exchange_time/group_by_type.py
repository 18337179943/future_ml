import pandas as pd
__Author__ = 'ZCXY'
import numpy as np
import os
import multiprocessing

def get_new_data(mouth_list):
    # base_path = '/mnt/zpool1/Data/future/future_price/'  
    base_path = '/mnt/DataServer/Data/future/future_price/'
    for moth in mouth_list:
        print(moth)
        for x in os.listdir(base_path + f'future_price{moth}'):
            if x[-4:] == '.txt':
                date = x[-12:-4]
                data_path = base_path + f'future_price{moth}/{x}'
                # save_path = '/mnt/zpool1/Data/future/group_by_type/'
                save_path = '/mnt/DataServer/share/future_datas_zc/group_by_type/'

                try:
                    data = pd.read_csv(data_path,  sep="\t")
                except:
                    continue
                data['TTIME'] = data['TTIME'].apply(lambda x : '{:0>6d}'.format(x))
                data['TTIME'] = pd.to_datetime(data['TTIME'], format='%H%M%S')
                data.dropna(subset=['TTIME'], inplace=True)
                data['TTIME'] = data['TTIME'].astype(str)
                data['TTIME'] = data['TTIME'].str[-8:]


                groups = data.groupby('CONTRACTID')
                contract_list = list(data['CONTRACTID'].drop_duplicates().values)

                for item in contract_list:
                    future = item[0:1] if item[1:2].isdigit() else item[0:2]
                    future_path = save_path + future
                    if not os.path.exists(future_path):
                        os.mkdir(future_path)
                    file_item = groups.get_group(item)
                    file_item.reset_index(drop='True', inplace=True)
                    res = file_item[['TDATE','CONTRACTID','TTIME','UPDATEMILLISEC','LASTPX','TQ','B1','BV1','S1','SV1','AVGPX','TM','OPENINTS','RISELIMIT','FALLLIMIT']]
                    name = item +'_' + date + '.csv'
                    res.to_csv(future_path + '/' + name, index=False, header=False)

def run_group_by_type():
    # iterm0 = ['202011', '202012']
    # iterm1 = [f'20210{i}' for i in range(1, 6)]
    iterm2 = [f'20210{i}' for i in range(7, 10)]
    iterm3 = [f'2021{i}' for i in range(10, 13)]
    # mouth_list = [iterm0, iterm1, iterm2, iterm3]
    mouth_list = [iterm2, iterm3]
    # mouth_list = [['202011', '202012'], ['202011'], ['202012']]
    
    pool = multiprocessing.Pool(len(mouth_list))
    
    for item in mouth_list:
        print(item)
        pool.apply_async(get_new_data(item))
    pool.close()
    pool.join()

if __name__ == "__main__":
    # 201912月我单独和过了，就不和了
    # mouth_list = [['201601','201602','201603','201604','201605','201606','201607','201608','201609','201610','201611','201612','201701','201702','201703',
    #                '201704','201705','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806',
    #                '201807','201808','201809','201810','201811','201812'],
    #               ['201901','201902','201903','201904','201905','201906','201907','201908','201909','201910','201911','202001','202002','202003',
    #                '202004','202005','202006','202007','202008','202009','202010','202011','202012','202101','202102','202103','202104','202105','202106']]
    run_group_by_type()