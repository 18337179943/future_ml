from lib2to3.pytree import Base
from pickletools import optimize
from turtle import forward
from matplotlib.colors import NoNorm
from sklearn.feature_selection import SelectKBest
import torch
from torch import nn
import sys, os
from m_base import *
sys_name = 'windows'
pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from backtesting import BackTester
from machine_learning.trainmodel import BaseModel
from datas_process.m_futures_factors import SymbolsInfo

# torch.manual_seed()

class MLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1, **kwargs) -> None:
        super(MLSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.out = nn.linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, c_n) = self.rnn(x)
        out = self.out(r_out[:, -1. :])
        return out


class DeepLearnClassification(BaseModel):
    
    def __init__(self, symbol=None, suffix='', model_name='lightgbm', need_test_set=1, zigzag=''):

        self.need_test_set = need_test_set
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
        self.col_n = int(re.findall("\d+",suffix)[-2])

    def get_datasets(self):
        '''获取训练集，验证集和测试集'''
        x_train, y_train, _ = self.get_xy(self.pa_train)
        x_valid, y_valid, _ = self.get_xy(self.pa_val)
        x_all, y_all, df_all = self.get_xy(self.pa_all)
        data_set = {'x_train': x_train,
                   'y_train': y_train,
                   'x_valid': x_valid,
                   'y_valid': y_valid}

        if self.need_test_set:
            x_test, y_test, _ = self.get_xy(self.pa_test)
            data_set.update({'x_test': x_test,
                             'y_test': y_test})
        
        datas_all = {
            'x': x_all,
            'y': y_all,
            'xy': df_all
        }
        return data_set, datas_all

    def get_xy(self, pa):
        '''获取训练数据，转换成三维矩阵''' 
        df = pd.read_csv(pa)
        df = df.set_index('datetime')
        df_o = df.copy()
        time_step = self.col_n # 相当于是正方形
        arr_x = []
        arr_y = df.pop('y').iloc[time_step:]
        for i in range(time_step, len(df)):
            arr_x.append(df.iloc[i-time_step:i].values)
        
        arr_x = np.array(arr_x)
        return arr_x, arr_y, df_o

    def set_params(self, params):
        '''设置模型参数
        params = {'input_size': , 'output_size': , 'hidden_size': 64, 'num_layers': ,
                  'epoch': , 'step_n': , 'batch_size': , 'loss_func_m': , 'lr': }
        '''
        if self.model is None:
            self.model = MLSTM(**params)

        self.params = params
        self.loss_func = self.get_loss_func(params['loss_func_m'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])

    def generate_model(self, model):
        if model is None:
            model = MLSTM()
        return model
        
    def seperate_datas(self):
        '''将数据分成训练集验证集和测试集 暂时没有用到'''
        train_time = self.sep_datetime['train_time']
        val_time = self.sep_datetime['val_time']
        test_time = self.sep_datetime['test_time']
        train_time = self.sep_datetime['train_datetime']
        data_y = pd.DataFrame(self.datas['y'])
        del self.datas['y']
        data_x = self.datas.copy()
        data_set = {'x_train': data_x[data_x.index < train_time],
                    'y_train': data_y[data_y.index < train_time],
                    'x_valid': data_x[(train_time <= data_x.index) & (data_x.index < val_time)],
                    'y_valid': data_y[(train_time <= data_x.index) & (data_x.index < val_time)],
                    'x_test': data_x[val_time <= data_x.index],
                    'y_test': data_x[val_time <= data_x.index],
                    'all_x': data_x,
                    'all_y': data_y}

        return data_set
            
    def get_loss_func(self, loss_func_m):
        '''获取损失函数'''
        if loss_func_m == 1:
            return nn.CrossEntropyLoss()

    def generate_x(self, step):
        '''生成x数据'''
        train_data_x = self.data_set['x_train'].iloc[step:]
        train_data_y = self.data_set['y_train'].iloc[step:]
        data_x = []
        data_y = []
        time_step = self.params['time_step']
        for i in range(self.params['batch_size']):
            data_x.append(train_data_x.iloc[i*time_step:time_step*(i+1)])
            data_y.append(train_data_y.iloc[i+time_step])
        x, y = torch.tensor(data_x), torch.tensor(data_y)
        return x, y

    def pnl(self, x, model=None, target_type='drawdown'):
        '''年化收益率/夏普比率'''
        if model==None:
            model = self.model
        y_pred = self.model_predict(x, model)
        y_pred.columns = ['y_pred']
        y_pred['datetime'] = x.index[self.params['time_step']:]
        # print(y_pred['datetime'].iloc[0], y_pred['datetime'].iloc[-1])
        # print('doneeeeeeeeeeeeeee')
        # input()
        bt = BackTester()
        _, annual_return = bt.all_contract_backtesting(self.symbol, y_pred['datetime'].iloc[0], y_pred['datetime'].iloc[-1], y_pred, target_type=target_type)
        return annual_return

    def model_predict(self, x, model=None):
        '''模型预测'''
        if model==None:
            model = self.model
        # torch_x = self.process_x(x.copy())
        output = model(x)
        y_predict = torch.max(output, 1)[1].data.numpy()
        y_predict = pd.DataFrame(x)
        时间还没有处理好---

        return y_predict

    def process_x(self, x: pd.DataFrame):
        '''将二维数据处理成三维数据'''
        x_li = []
        time_step = self.params['time_step']
        for i in range(len(x)-time_step):
            x_li.append(x.iloc[i, i+time_step].values.tolist())
        torch_x = torch.tensor(x_li)
        return torch_x
        
    def lstm_train(self):
        '''模型训练'''
        batch_size = self.params['batch_size']
        for i in range(self.params['epoch']):
            for step in range(0, len(self.data_set['x_train'])-batch_size, batch_size):
                b_x = torch.tensor(self.data_set['x_train'][step:step+batch_size])
                b_y = torch.tensor(self.data_set['y_train'].values[step:step+batch_size])
                # b_x, b_y = self.generate_x(step)
                output = self.model(b_x)
                loss = self.loss_func(output, b_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            



        
                





                











