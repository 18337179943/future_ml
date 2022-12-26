import numpy as np
import pandas as pd
__Author__ = 'ZCXY'
import os
import pymysql
from pandas.core.frame import DataFrame
import warnings
warnings.filterwarnings("ignore")


# connect = pymysql.connect(host='rm-uf68ez0445d101yimvo.mysql.rds.aliyuncs.com',
# port=3306,
# db='tldata_db',
# user='liunian',
# password='IsFlo5R82lhdwMtmu24Y')

connect = pymysql.connect(host='rm-uf696j1mdpk2ci0u3zo.mysql.rds.aliyuncs.com',
port=3306,
db='tldata_db',
user='hq_caiji',
password='Qy7P7nOUxfaHFPEF1GcH')



# 数据精度量化压缩
def reduce_mem_usage(df):
    # 处理前 数据集总内存计算
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    # 遍历特征列
    for col in df.columns:
        # 当前特征类型
        col_type = df[col].dtype
        # 处理 numeric 型数据
        if col_type != object:
            c_min = df[col].min()  # 最小值
            c_max = df[col].max()  # 最大值
            # int 型数据 精度转换
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            # float 型数据 精度转换
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # 处理 object 型数据
        else:
            df[col] = df[col].astype('category')  # object 转 category
    
    # 处理后 数据集总内存计算
    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# 读取sql
def query_sql_df(sql):
    cursor = connect.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(data , columns = columns)
    need_astype = df.select_dtypes(include=[np.object]).columns
    if not need_astype.empty:
        need_astype = np.delete(need_astype, [0, 1, 2])  # 删除'date', 'object', 'symbol'
        if 'last_deli_date' in need_astype:
            need_astype = list(need_astype)
            del need_astype[-1]
        #     need_astype = np.delete(need_astype, [0, 1, 2])   # 删除'last_deli_date', 'maincon', 'smaincon'
        #     df['maincon'] = df['maincon'].apply(lambda x: ord(x))
        #     df['smaincon'] = df['smaincon'].apply(lambda x: ord(x))
        df[need_astype] = df[need_astype].astype('float')
        # df[need_astype] = reduce_mem_usage(df[need_astype])
    return df

# 查看每列缺失值个数
def check_missing_column(data: DataFrame):
    '''查看每列缺失值个数'''
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    print("缺失值概况如下：\n{}\n".format(missing))

# 查看含有缺失值的行(指定列)
def check_missing_row(df: DataFrame, col=None):
    '''
    col = ['mem_vol', 'mem_long_oi', 'mem_short_oi']
    '''
    if not col:
        print(df[df.isnull().T.any()])
    else:
        print(df[df[col].isnull().T.any()][col])

# 转换为双重索引
def transform_multiindex(df: DataFrame):
    df = df.set_index(['date', 'object'])  # 双重索引 -- 时间，标的
    return df

# 因子pivot
def transform_pivot(df: DataFrame, factor):
    df = df.pivot(index='date', columns='object', values=factor)
    return df

# 读取数据
def query_data(func, filename, type='rawdata'):
    '''
    rawdat = RawData()
    query(rawdat.query_raw_volprice_data, 'raw_volprice')

    type: 'rawdata' 原始数据/ 'factor' 因子
    '''
    if not os.path.exists('data/{}/{}.csv'.format(type, filename)):
        df = func()
        df.to_csv('data/{}/{}.csv'.format(type, filename), index=False)
        print(f"{type}/{filename}...已下载完成/计算完毕")
        return df
    else:
        df = pd.read_csv('data/{}/{}.csv'.format(type, filename))
        print(f"{type}/{filename}...已读取完毕")
        return df


def check_index(df_more, df_less):
    '''检查df_more中含有，但df_less中没有的index'''
    set_more = set(transform_multiindex(df_more).index)
    set_less = set(transform_multiindex(df_less).index)
    print("前者中含有，但后者中没有的索引")
    print(set_more.difference(set_less))
    # for i in df_more.index:
    #     if i not in df_less.index:
    #         print(i)

# debug专用
def debug(df: DataFrame):
    from datetime import date
    if df['object'].values[0]=='SC':
        df = df[(df['object']=='SC') & (df['date']>date(2021,3,15)) & (df['date']<date(2021,5,10))]
        # df = df[(df['date']>date(2017,11,1)) & (df['date']<date(2018,2,1))][df['to_volume']<1]
        # df = df[df['to_volume']<1]
        print(df)
        input()
        # print(df.head(20).append(df.tail(20)))
        # print(df['date'].between('20170101', '20180101'))
        # print(df[df['date']>datetime.date(2017,1,1)])
