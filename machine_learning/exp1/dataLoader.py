import json
import os

import jieba
import numpy as np


# 根据某列所有字符串列表，产生分词编码表
def generate_encode_dictionary_v1(sentents):
    all_words = set()
    for s in sentents:
        all_words.update(jieba.lcut(s))
    return sorted(all_words)


# 根据某列所有字符串列表词频top N，产生分词编码表
def generate_encode_dictionary_v2(sentents, use_jieba=False):
    stat_words = {}
    if use_jieba:
        for s in sentents:
            for j in jieba.lcut(s):
                stat_words[j] = stat_words[j] + 1 if j in stat_words else 0
    else:
        for s in sentents:
            stat_words[s] = stat_words[s] + 1 if s in stat_words else 0
    # 取频率最高的前50个词作为编码字典
    sorted_words = sorted(stat_words.items(), key=lambda x: x[1], reverse=True)[:50]
    print(sorted_words)
    return [w[0] for w in sorted_words]


# 根据分词编码表对某个字符串编码
def gen_encode_1(all_words, str):
    return [jieba.lcut(str).count(w) for w in all_words]


# 根据分词编码表对某个字符串编码, 每100个合计计数一次
def gen_encode_v2(all_words, str):
    return [sum([w.count(s) for s in jieba.lcut(str)]) for w in all_words[::100]]


# 按照encode_column_map中的列产生分词编码字典总表，并保存到文件中
def gen_all_encode_dictionary(path, encode_column_map, data):
    all_words_dict = {}
    for column, name in encode_column_map.items():
        '''
        if name in ["详细地址"]:
            all_words_dict[name] = generate_encode_dictionary_v2(data[:, column], True)
        else:
            all_words_dict[name] = generate_encode_dictionary_v2(data[:, column])
        '''
        all_words_dict[name] = generate_encode_dictionary_v1(data[:, column])
    with open(path, mode='w', encoding='utf8') as f:
        json.dump(all_words_dict, f)
    return all_words_dict


# 从文件加载分词编码字典总表，返回字典
def load_all_encode_dictionary(path):
    with open(path, mode='r', encoding='utf8') as f:
        return json.load(f)


# 加载最原始的数据，并按照逗号分隔，确保列对齐，返回的时间都是原始字符串，不能用于模型处理
def load_raw_data(path):
    header = []
    x = []
    y = []
    columns = 0
    with open(path, mode='r', encoding='utf8') as f:
        header = f.readline().strip().split(",")
        columns = len(header)

        for line in f:
            sample = line.strip().split(",")
            columns_sample = len(sample)

            # 比表头多处的列数就是地址当前记录地址列的长度
            columns_diff = columns_sample - columns
            # 价格是模型的输出量y,从特征中去除
            new_sample = [s for s in sample[6 + columns_diff + 1:]]
            y.append(sample[6 + columns_diff])
            new_sample = [s for s in sample[3 + columns_diff + 1:6 + columns_diff]] + new_sample

            # 取地址的多列合并成为一个列
            if columns_diff == 0:
                new_sample = sample[3:3 + columns_diff + 1] + new_sample
            else:
                new_sample = ["".join(sample[3:3 + columns_diff + 1]).strip('"')] + new_sample

            # 地址列前面的列放在前面
            new_sample = [s for s in sample[0:3]] + new_sample

            x.append(new_sample)

    return x, y


# 加载数据为人工智能模型可以训练的数字序列
def load_horse_rental_data():
    encode_dict_path = "encode_dictionary.json"
    encode_column_map = {
        1: "租房网站名称",
        2: "小区",
        3: "详细地址",
        4: "城市",
        5: "区",
        10: "朝向"
    }

    # 不做处理可以忽略的列,添加到这个列表中
    ignore_column = [0, 3]

    raw_X, raw_y = load_raw_data("../dataset/中国租房信息数据集.csv")
    records = np.array(raw_X)

    # 获取编码字典
    encode_dict = {}
    if os.path.exists(encode_dict_path):
        encode_dict = load_all_encode_dictionary(encode_dict_path)
    else:
        encode_dict = gen_all_encode_dictionary(encode_dict_path, encode_column_map, records)

    # 开始数字编码
    new_X = []
    new_y = [float(y) for y in raw_y]
    for d in records:
        record = []
        for j in range(len(d)):
            # 忽略的列不做编码
            if j in ignore_column:
                continue

            # 需要编码的字符串列开始通过分词编码，不需要编码的列转换字符串为数字
            if j in encode_column_map.keys():
                record += gen_encode_v2(encode_dict[encode_column_map[j]], d[j])
            else:
                # 字符串串中含有.的转换为浮点点数，其他的转换为整数
                if '.' in d[j]:
                    record.append(float(d[j]))
                else:
                    record.append(int(d[j]))

        new_X.append(record)

    return new_X, new_y
