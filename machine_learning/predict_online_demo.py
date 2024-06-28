"""
Author: jhzhu
Date: 2024/6/21
Description: 
"""
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*unknown categories.*during transform.*")

model = joblib.load('./page_code/model/adaboost.joblib')

input_data = {
    '租房网站名称': '房天下',  # ['房天下', '58同城', '赶集网']
    '城市': '深圳',  # ['深圳', '上海', '北京']
    '区': '南山',
    '小区': '前海时代',
    '室': 2,
    '卫': 2,
    '厅': 1,
    '面积': 90,
    '朝向': '南',  # value choices: ['南','东西','西南','东','西','东北','西北','北']
    '所属楼层': 4,
    '总楼层': 26,
    '是否有阳台': 1,
    '信息发布人类型': 0,
    '是否有床': 1,
    '是否有衣柜': 1,
    '是否有沙发': 1,
    '是否有电视': 1,
    '是否有冰箱': 1,
    '是否有洗衣机': 1,
    '是否有空调': 0,
    '是否有热水器': 1,
    '是否有宽带': 1,
    '是否有燃气': 1,
    '是否有暖气': 1,
    'lng': 113.9117133,
    'lat': 22.53148739,
    '最近学校距离': 1542,  # [-1,3842]
    '周边学校个数': 3,
    '最近医院距离': 3755, # [-1,401]
    '周边医院个数': 2
}

input_df = pd.DataFrame([input_data])
# single price
predicted_price = model.predict(input_df)
# total price
total_price = predicted_price[0] * input_data['面积']
print(f'Predicted price of one month: {int(total_price)}￥')
