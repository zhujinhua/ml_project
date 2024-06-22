"""
Author: jhzhu
Date: 2024/6/20
Description: 
"""
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, \
    BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns

ENCODER_COLUMNS = ['租房网站名称', '小区', '城市', '区', '朝向']

numeric_features = ['面积', 'lng', 'lat', '所属楼层', '总楼层', '最近学校距离', '周边学校个数', '最近医院距离',
                    '周边医院个数']

categorical_features = ['租房网站名称', '小区', '城市', '区', '室', '卫', '厅', '朝向',
                        '是否有阳台', '信息发布人类型', '是否有床', '是否有衣柜', '是否有沙发',
                        '是否有电视', '是否有冰箱', '是否有洗衣机', '是否有空调', '是否有热水器',
                        '是否有宽带', '是否有燃气', '是否有暖气']


# FILL_VALUE = ['南','东西','西南','东','西','东北','西北','北']


def plot_correlation_matrix(df, feature_importances, top_n=10):
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    # Sort the features by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    top_features = feature_importances.head(top_n)['Feature'].values
    correlation_matrix = df[top_features].corr()
    sns.set(style='white')
    plt.rcParams['font.sans-serif'] = ['Yuanti SC']
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                          annot_kws={"size": 12})
    plt.title('Top 10 Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.show()


def custom_adjusted_r2(y_true, y_pred, **kwargs):
    if 'x_column' not in kwargs['kwargs']:
        return 0
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    return 1 - (1 - r2) * (len(y_pred) - 1) / (len(y_pred) - kwargs['kwargs']['x_column'] - 1)


def custom_error_percentage(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        return round(1 - np.sum(np.abs(y_true.values.flatten() - y_pred)) / np.sum(y_true), 3)[0]
    else:
        return round(1 - np.sum(np.abs(y_true - y_pred)) / np.sum(y_true), 3)


def custom_error_percentage_avg(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    return np.round(np.mean(1 - np.abs(y_true - y_pred) / y_true), 3)


def evaluate_predict_result(x, y_true, y_pred):
    result_dict = dict()
    result_dict['mean_absolute_error'] = round(mean_absolute_error(y_true=y_true, y_pred=y_pred), 3)
    result_dict['median_absolute_error'] = round(median_absolute_error(y_true=y_true, y_pred=y_pred), 3)
    result_dict['root_mean_squared_error'] = round(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 3)
    result_dict['r2'] = round(r2_score(y_true=y_true, y_pred=y_pred), 3)
    result_dict['adjusted_r2'] = round(custom_adjusted_r2(y_true, y_pred, kwargs={'x_column': x.shape[1]}), 3)
    result_dict['accuracy'] = round(custom_error_percentage(y_true, y_pred), 3)
    result_dict['avg accuracy'] = round(custom_error_percentage_avg(y_true, y_pred), 3)
    return result_dict


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
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.show()


def visualize_shap_values(model, X_train, X_test):
    explainer = shap.KernelExplainer(model=model.predict, data=shap.sample(X_train, 3000))
    sample_X_test_pd = shap.sample(X_test, 3000)
    shap_values = explainer.shap_values(sample_X_test_pd, nsamples=100)

    pd_shap_values = pd.DataFrame(shap_values, columns=X_train.columns)
    print('pd_shap_values.abs().mean():', pd_shap_values.abs().mean())
    shap.summary_plot(shap_values, sample_X_test_pd)


def zero_index_features(df, columns_to_reindex):
    for column in columns_to_reindex:
        df[column] = pd.factorize(df[column])[0]


def get_column_transformer(encoded_columns_name):
    return ColumnTransformer([('encoder', OneHotEncoder(drop='first'), encoded_columns_name)], remainder='passthrough')


def plot_models_predict_result(x_labels, y_values, y_label):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    bars = plt.bar(x_labels, y_values, color=sns.color_palette("coolwarm", len(x_labels)))
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

    plt.title('Model Training Performance Comparison', fontsize=20, fontweight='bold')
    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


rent_house_df = pd.read_csv('../dataset/中国租房信息数据集.csv')
# drop NULL Value, feature link, 详细地址 for not useful
filtered_df = rent_house_df.drop(columns=['link', '详细地址']).dropna()
# -1 represent to max
# filtered_df.loc[(filtered_df['最近学校距离'] == -1), '最近学校距离'] = 4000
# filtered_df.loc[(filtered_df['最近医院距离'] == -1), '最近医院距离'] = 5000
# filter data 面积 <5, 面积/室 < 3
filtered_df = filtered_df[(filtered_df['面积'] >= 5) & (filtered_df['面积'] / filtered_df['室'] >= 3)]
filtered_df.loc[(filtered_df['所属楼层'] > filtered_df['总楼层']), '总楼层'] = filtered_df['所属楼层']
filtered_df['价格'] = filtered_df['价格'] / filtered_df['面积']
zero_index_features(filtered_df, ENCODER_COLUMNS)
X = filtered_df.loc[:, filtered_df.columns != '价格']
y = filtered_df.loc[:, '价格']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# applies transformers to columns: discrete feature: one-hot encode, continuous feature: standardization
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('category', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ]
)
# grid search to train the hyper parameters
'''
estimator = LinearRegression()
estimator = SVR()
estimator = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
estimator = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.2)
estimator = AdaBoostRegressor(estimator=DecisionTreeRegressor(), n_estimators=500, learning_rate=0.05, random_state=42)


Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=3)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", np.sqrt(-grid_search.best_score_))
'''
# estimator list to try different models
estimator_dict = {
    'Linear Regression': LinearRegression(),
    'SVM': SVR(kernel='linear', C=10, epsilon=0.2),
    'Decision Tree': DecisionTreeRegressor(max_depth=35, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=35),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.2),
    'Bagging': BaggingRegressor(n_estimators=200, random_state=42),
    'AdaBoost': AdaBoostRegressor(estimator=DecisionTreeRegressor(), n_estimators=500, learning_rate=0.05,
                                  random_state=42)
}
mae_dict = {}
accuracy_dict = {}
training_time = {}
for key, estimator in estimator_dict.items():
    start = time.localtime()
    # define the training pipeline: preprocess, estimate model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', estimator)
    ])
    pipeline.fit(X_train, y_train)
    # joblib.dump(pipeline, filename='../output/adaboost.joblib', compress=6)
    train_end = time.localtime()
    y_pred = pipeline.predict(X_test)
    pred_end = time.localtime()
    train_time = (time.mktime(train_end) - time.mktime(start))
    print('%s training cost %ss, predict cost %ss' % (key, train_time,
                                                      (time.mktime(pred_end) - time.mktime(train_end))))
    result = evaluate_predict_result(X_test, y_test, y_pred)
    mae_dict[key] = result['mean_absolute_error']
    accuracy_dict[key] = result['avg accuracy']
    training_time[key] = train_time
    print('%s: %s' % (key, result))
plot_models_predict_result(list(mae_dict.keys()), list(mae_dict.values()), 'Mean Absolute Error(MAE)')
plot_models_predict_result(list(accuracy_dict.keys()), list(accuracy_dict.values()), 'Average Accuracy')
plot_models_predict_result(list(training_time.keys()), list(training_time.values()), 'Training Time(s)')

# plot the feature importance based on random forest
'''
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
rf.fit(X_train, y_train)
importance = rf.feature_importances_
plot_feature_importance(importance, X_train)
# visualize_shap_values(rf, X_test, X)
rf_pred = rf.predict(X_test)
output = evaluate_predict_result(X_test, y_test, rf_pred)
print(output)
'''
