"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ENCODER_COLUMNS = ['link', '详细地址', '租房网站名称', '城市', '小区', '区', '朝向']


def zero_index_features(df, columns_to_reindex):
    for column in columns_to_reindex:
        df[column] = pd.factorize(df[column])[0]


def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    sns.set(style='white')
    plt.rcParams['font.sans-serif'] = ['Yuanti SC']
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                          annot_kws={"size": 12})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.show()


rent_house_df = pd.read_csv('../dataset/中国租房信息数据集.csv')
rent_house_df.dropna()
zero_index_features(rent_house_df, ENCODER_COLUMNS)
plot_correlation_matrix(rent_house_df)
