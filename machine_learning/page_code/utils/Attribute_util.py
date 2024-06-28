import pandas as pd
import numpy as np


aim_val = ['租房网站名称','城市','区']

xq_lng_lat = ['小区','lng','lat']

def _read_data(path=None):
    """
        读取数据
            path：文件路径
    """
    df = pd.read_csv(path)
    return df


import pandas as pd
import numpy as np


aim_val = ['租房网站名称','城市','区']

xq_lng_lat = ['小区','lng','lat']

def _read_data(path=None):
    """
        读取数据
            path：文件路径
    """
    df = pd.read_csv(path)
    return df


def construct_attribute(file_name = "./../data/中国租房信息数据集.csv"):
    """
        构建属性信息
        参数：file_name：文件路径
        返回：
            attribute：属性字典
            lng_lat：小区经纬度
            q_xq：区映射小区
            cq：城市映射区
    """
    original_data = _read_data(path=file_name)
    attribute = {}
    # ,'lng','lat'
    lng_lat = {}
    # 区：小区
    q_xq = {}
    # 城市：区
    cq = {}
    for idx,column in enumerate(original_data.columns):
        temp=None
        if aim_val.count(column)==1:
            temp = np.unique(original_data[column].to_numpy())
            attribute[column] = temp
        if column == '小区':
            xq = original_data.drop_duplicates(subset=[column])
            attribute[column] = xq[column].to_numpy()
            for name,lng,lat in xq[xq_lng_lat].to_numpy():
                lng_lat[name] = {'lng':lng,'lat':lat}
        if column == '区':
            for q in temp:
                q2xq = original_data[original_data['区']==q].drop_duplicates(subset=['小区'])['小区']
                q_xq[q] = q2xq
        if column == '城市':
            for q in temp:
                c2q = original_data[original_data['城市']==q].drop_duplicates(subset=['区'])['区']
                cq[q] = c2q
    return attribute,lng_lat,q_xq,cq
