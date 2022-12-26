from asyncore import file_dispatcher
import pandas as pd
__Author__ = 'ZCXY'
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def makedir(pa):
    if not os.path.exists(pa):
        os.makedirs(pa)
    return pa

class GenerateKline():
    '''服务器上k线合成'''
    def __init__(self) -> None:
        self.load_pa = '/mnt/DataServer/Data/future/kline_1min/'
        self.save_pa = makedir('/mnt/DataServer/share/future_datas_zc/datas_1min/')
        self.year_li = ['2016', '2017', '2018', '2019', '2020']
        self.month_li = [f'0{i}' if i < 10 else str(i) for i in range(1, 13)]
        self.symbol_li = ['AG', 'AL', 'AP', 'BU', 'C', 'CF', 'CS', 'CU', 'FG', 'HC', 'I', 'IC', 'IF', 'IH', 
                          'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P', 'PB', 'PM', 'PP', 'RB', 'RM', 'RU', 
                          'SC', 'SF', 'SM', 'SN', 'SR', 'TA', 'V', 'Y', 'ZC', 'ZN']
        self.save_group_by_type_pa = '/mnt/DataServer/share/future_datas_zc/group_by_type/'

    def get_symbol_k_line(self, sy):
        '''获取和拼接k线''' 
        for year in self.year_li:
            for month in self.month_li:
                try:
                    pa = f'{self.load_pa}{year}/{month}/{sy}/'
                    file_li = os.listdir(pa)
                    save_pa = makedir(f'{self.save_group_by_type_pa}{sy}/')
                    for i in file_li:
                        pd.read_csv(f'{pa}{i}').to_csv(f'{save_pa}{i}', index=False)
                    # [shutil.copy(f'{pa}{i}', save_pa) for i in file_li]
                    print(sy, year, month)
                except:
                    print('wrong: ', f'{self.load_pa}{year}/{month}/{sy}/')

    def get_symbol_k_line_mp(self):
        self.multiprocess(self.get_symbol_k_line, self.symbol_li)

    def combine_symbol_k_line(self, symbol):
        '''将每个品种合约所有k线拼接在一起'''
        save_pa = makedir(f'{self.save_pa}{symbol}/')
        all_contract_file_li = os.listdir(f'{self.save_group_by_type_pa}{symbol}/')
        contract_li = [contract.split('_')[0] for contract in all_contract_file_li]
        contract_li = list(set(contract_li))
        for contract in contract_li:
            df_li = []
            contract_file_li = list(filter(lambda x: contract in x, all_contract_file_li))
            for file_pa in contract_file_li:
                df_li.append(pd.read_csv(f'{self.save_group_by_type_pa}{symbol}/{file_pa}'))
            df_contract = pd.concat(df_li, ignore_index=True)
            df_contract.to_csv(f'{save_pa}{contract}.csv', index=False)
            print(contract)

    def combine_symbol_k_line_mp(self):
         self.multiprocess(self.combine_symbol_k_line, self.symbol_li)

    def multiprocess(self, func, li):
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:  # max_workers=10
            executor.map(func, li)
        print(func.__name__, 'done.')
        return 

    def main(self):
        self.get_symbol_k_line_mp()
        # self.get_symbol_k_line('AP')
        print('start combine_symbol_k_line_mp')
        self.combine_symbol_k_line_mp()
        # self.combine_symbol_k_line('AP')


def run_generatekline():
    gkl = GenerateKline()
    gkl.main()


if __name__ == '__main__':
    run_generatekline()



