import pandas as pd
__Author__ = 'ZCXY'
import os, sys
import tarfile
import re, shutil


def un_zip(tar_pa, save_pa):
    tar_li = os.listdir(tar_pa)
    if not os.path.exists(save_pa):
        os.makedirs(save_pa)
    for tar_name in tar_li:
        print(tar_name)
        tar_file = tarfile.open(f'{tar_pa}{tar_name}')
        target_name = tar_name.split('.')[0]
        tar_file.extractall(f'{save_pa}{target_name}')

def get_sy(contract):
    re_func = re.compile(r'(\d+|\s+)')
    sy_li = re_func.split(contract)
    return sy_li

def move_datas(date_adj, load_pa, save_pa):
    folder_li = os.listdir(load_pa)
    for folder_name in folder_li:
        market_date = re.findall("\d+", folder_name)[0]
        if market_date in date_adj:
            print(market_date)
            contract_li = os.listdir(f'{load_pa}{folder_name}')
            for contract in contract_li:
                sy_li = get_sy(contract.upper())
                if len(sy_li) == 3:
                    symbol = sy_li[0]
                    st = '2' if len(sy_li[1]) < 4 else ''
                    contract_name = f'{symbol}{st}{sy_li[1]}'
                    # contract_name = contract.split('.')[0].upper()
                    save_contract_name = f'{contract_name}_{market_date}.csv'
                    # print(symbol, contract)
                    # print(f'{load_pa}{folder_name}/{contract}')
                    # print(f'{save_pa}{symbol}/{save_contract_name}')
                    # print('---------')
                    # input()
                    try:
                        shutil.copyfile(f'{load_pa}{folder_name}/{contract}', f'{save_pa}{symbol}/{save_contract_name}')
                    except:
                        print(f'{save_pa}{symbol}/{save_contract_name}')
                        continue

def run_qh_marketdata():
    tar_pa = '/mnt/DataServer/share/qh_marketdata/2021_marketdata/'
    save_tick_pa = '/mnt/DataServer/share/future_datas_zc/qh_marketdata/2021_marketdata/'
    load_pa = save_tick_pa
    move_tick_pa = '/mnt/DataServer/share/future_datas_zc/group_by_type/'
    date_adj = ['20210702','20210715','20210720','20210722','20210723','20210809','20210810',
                '20210811','20210812','20210824','20210826','20211012','20211018','20211105',
                '20211117','20211202','20211203','20211206','20211207','20211213','20211216','20211222','20211223','20211229']
    # un_zip(tar_pa, save_tick_pa)
    move_datas(date_adj, load_pa, move_tick_pa)

if __name__ == '__main__':
    
    run_qh_marketdata()