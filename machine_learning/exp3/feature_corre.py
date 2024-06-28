"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ENCODER_COLUMNS = ['link', '详细地址', '租房网站名称', '城市', '小区', '区', '朝向']
ENCODER_COLUMNS = ['租房网站名称', '城市', '小区', '区', '朝向']


def zero_index_features(df, columns_to_reindex):
    for column in columns_to_reindex:
        df[column] = pd.factorize(df[column])[0]


def plot_correlation_matrix(df):
    # get the dataframe corr matrix
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


def plot_feature_importance(importances, X):
    features = X.columns

    indices = np.argsort(importances)[::-1]
    plt.rcParams['font.sans-serif'] = ['Yuanti SC']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(16, 12))

    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(len(features)), np.array(features)[indices], rotation=45, ha='right')

    plt.title("Feature Importance", fontsize=20, fontweight='bold')
    plt.xlabel('Features', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=14, fontweight='bold')
    for i, v in enumerate(importances[indices]):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.show()


rent_house_df = pd.read_csv('../dataset/中国租房信息数据集.csv')
# rent_house_df.dropna()
# zero_index_features(rent_house_df, ENCODER_COLUMNS)
# plot_correlation_matrix(rent_house_df)

# plot the feature importance based on random forest
filtered_df = rent_house_df.drop(columns=['link', '详细地址']).dropna()
filtered_df = filtered_df[(filtered_df['面积'] >= 5) & (filtered_df['面积'] / filtered_df['室'] >= 3)]
filtered_df.loc[(filtered_df['所属楼层'] > filtered_df['总楼层']), '总楼层'] = filtered_df['所属楼层']
# filtered_df['价格'] = filtered_df['价格'] / filtered_df['面积']
zero_index_features(filtered_df, ENCODER_COLUMNS)
X = filtered_df.loc[:, filtered_df.columns != '价格']
y = filtered_df.loc[:, '价格']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
rf.fit(X_train, y_train)
importance = rf.feature_importances_
plot_feature_importance(importance, X_train)
# visualize_shap_values(rf, X_test, X)

