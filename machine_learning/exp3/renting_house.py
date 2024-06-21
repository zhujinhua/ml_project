"""
Author: jhzhu
Date: 2024/6/20
Description: 
"""
"""
Author: jhzhu
Date: 2024/6/16
Description: 
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
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

categorical_features = ['租房网站名称', '小区', '城市', '区', '室', '卫', '厅', '朝向']


# FILL_VALUE = ['南','东西','西南','东','西','东北','西北','北']

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


def plot_feature_importance(model, X):
    importances = model.feature_importances_
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


def adjusted_pred(y_pred, X_test):
    y_pred = pd.Series(y_pred)
    return y_pred * X_test['面积'].reset_index(drop=True)


def plot_models_predict_result(x_labels, y_values):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    bars = plt.bar(x_labels, y_values, color=sns.color_palette("viridis", len(x_labels)))
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

    plt.title('Model Training Performance Comparison', fontsize=20, fontweight='bold')
    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Absolute Error(MAE)', fontsize=14, fontweight='bold')
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
# filtered_df = filtered_df[filtered_df['所属楼层'] <= filtered_df['总楼层']]
filtered_df['价格'] = filtered_df['价格'] / filtered_df['面积']
zero_index_features(filtered_df, ENCODER_COLUMNS)
X = filtered_df.loc[:, filtered_df.columns != '价格']
y = filtered_df.loc[:, '价格']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
y_test = y_test.reset_index(drop=True)
y_test_adjusted = y_test * X_test['面积'].reset_index(drop=True)
#
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('category', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ]
)

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

estimator_dict = {
    'Linear Regression': LinearRegression(),
    'SVM': SVR(kernel='linear', C=10, epsilon=0.2),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.2),
    'AdaBoost': AdaBoostRegressor(estimator=DecisionTreeRegressor(), n_estimators=500, learning_rate=0.05,
                                  random_state=42)
}
result_dict = {}
for key, estimator in estimator_dict.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', estimator)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = adjusted_pred(y_pred, X_test)
    result = evaluate_predict_result(X_test, y_test_adjusted, y_pred)
    result_dict[key] = result['mean_absolute_error']
    print('%s: %s' % (key, result))
plot_models_predict_result(list(result_dict.keys()), list(result_dict.values()))


'''
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
rf.fit(X_train, y_train)
plot_feature_importance(rf, X_train)
# visualize_shap_values(rf, X_test, X)
rf_pred = rf.predict(X_test)
# update predict value
y_test = y_test.reset_index(drop=True)
rf_pred = pd.Series(rf_pred)

rf_pred_adjusted = rf_pred * X_test['面积'].reset_index(drop=True)

result = evaluate_predict_result(X_test, y_test_adjusted, rf_pred_adjusted)
print(result)
'''

