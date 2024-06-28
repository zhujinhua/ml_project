"""
Author: jhzhu
Date: 2024/6/25
Description: 
"""
data = []
file_name = "../dataset/中国租房信息数据集.csv"
with open(file=file_name, mode='r', encoding='utf8') as f:
    keys = f.readline().strip().split(',')
    keys = keys[1:3] + keys[-29:]
    for idx, line in enumerate(f):
        line = line.strip()
        # 删除地址详情列
        if line:
            line = line.split(',')
            line = line[1:3] + line[-29:]
        data.append(line)
import numpy as np

data = np.array(data, dtype=object)
data_dict = {keys[i]: data[:, i] for i in range(data.shape[1])}
# %%
# 条形图数据初始化
models = []
mse_scores = []  # 均方误差（MSE）分数
mae_scores = []  # 平均绝对误差（MAE）分数
average_accuracy_scores = []  # 平均绝对误差（MAE）分数
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

df = pd.DataFrame(data_dict)

# 特征和目标变量
X = df.drop(columns=['价格'])
y = df['单价'] = pd.to_numeric(df['价格']) / pd.to_numeric(df['面积'])
# y = df['价格']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数值特征
numeric_features = ['面积', 'lng', 'lat', '最近学校距离', '周边学校个数', '最近医院距离', '周边医院个数']

# 类别特征
categorical_features = ['租房网站名称', '小区', '城市', '区', '室', '卫', '厅', '朝向',
                        '所属楼层', '总楼层', '是否有阳台', '信息发布人类型', '是否有床', '是否有衣柜', '是否有沙发',
                        '是否有电视', '是否有冰箱', '是否有洗衣机', '是否有空调', '是否有热水器', '是否有宽带',
                        '是否有燃气', '是否有暖气']

# 数值特征处理：缺失值填补+标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()), ('pca', PCA(n_components=5))
])

# 类别特征处理：缺失值填补+独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# 将所有处理步骤整合到 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def custom_error_percentage_avg(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    return np.round(np.mean(1 - np.abs(y_true - y_pred) / y_true), 3)


def MLtrain(regressorModel):
    # 将预处理步骤和回归模型整合到 Pipeline 中
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressorModel)])

    # 训练模型
    model.fit(X_train, y_train)

    # 预测和评估
    single_pred = model.predict(X_test)
    y_pred = single_pred * pd.to_numeric(X_test['面积'])
    mse = mean_squared_error(y_test * pd.to_numeric(X_test['面积']), y_pred)
    mae = mean_absolute_error(y_test * pd.to_numeric(X_test['面积']), y_pred)
    av_acc = round(custom_error_percentage_avg(y_test, single_pred), 3)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # 条形图数据

    mse_scores.append(mse)
    mae_scores.append(mae)
    average_accuracy_scores.append(av_acc)
    return model

# 使用线性回归得到模型
from sklearn.linear_model import LinearRegression
models.append('线性回归')
linearRegressionModel = MLtrain(LinearRegression())

# 随机森林回归
from sklearn.ensemble import RandomForestRegressor
models.append('随机森林回归')
randomForestRegressorModel = MLtrain(RandomForestRegressor())

# 支持向量机
from sklearn.svm import SVR

models.append('支持向量机')
SVRModel = MLtrain(SVR())

# K近邻回归
from sklearn.neighbors import KNeighborsRegressor
models.append('K近邻回归')
kNeighborsRegressorModel = MLtrain(KNeighborsRegressor(n_neighbors=5))

# 决策树回归
from sklearn.tree import DecisionTreeRegressor
models.append('决策树回归')
decisionTreeRegressorModel = MLtrain(DecisionTreeRegressor())


# 梯度提升回归
from sklearn.ensemble import GradientBoostingRegressor
models.append('梯度提升回归')
gradientBoostingRegressorModel = MLtrain(GradientBoostingRegressor())