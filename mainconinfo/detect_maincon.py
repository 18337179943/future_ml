import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import pandas as pd
__Author__ = 'ZCXY'
from m_base import read_json, save_json
from mainconinfo.maincon import run_MainCon

'''需要在dataload里将start更改一下'''

def detect_maincon():
    '''判断当前交易合约是否为主力合约，选出非主力合约'''
    pa = f'{pa_prefix}/datas/mainconinfo/'
    js_pa = pa+'simulation_contract.json'
    df_maincon = pd.read_csv(pa+'df_symbol_chg_maincon.csv')
    simultion_js = read_json(js_pa)
    df_maincon = df_maincon[df_maincon['maincon']==True]
    symbols_li = [i.upper() for i in simultion_js['symbols']]
    contracts_li = [i.split('.')[0] for i in simultion_js['contracts']]
    maincon_li = [df_maincon[df_maincon['object']==sy]['symbol'].iloc[-1] for sy in symbols_li]
    old_contracts = list(filter(lambda x: x not in maincon_li, contracts_li))
    new_contracts = list(filter(lambda x: x not in contracts_li, maincon_li))
    simultion_js['old_contracts'] = old_contracts
    simultion_js['new_contracts'] = new_contracts
    # save_json(simultion_js, js_pa)
    print('date: ', df_maincon['date'].iloc[-1])
    print("old_contracts: ", old_contracts)
    print("new_contracts: ", new_contracts)
    return old_contracts, new_contracts

def del_maincon_file():
    pa = f'{pa_prefix}/datas/mainconinfo/'
    li = os.listdir(pa)
    li = list(filter(lambda x: 'csv' in x, li))
    [os.remove(pa+i) for i in li]

def run_dectect_maincon():
    del_maincon_file()
    try:
        run_MainCon()
    except:
        pass
    run_MainCon()
    detect_maincon()
    
if __name__ == "__main__":
    run_dectect_maincon()
    
