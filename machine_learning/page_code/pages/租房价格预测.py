import streamlit as st
import utils.load_utils as load_model
import utils.Attribute_util as load_attribute
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, message=".*unknown categories.*during transform.*")

# 页面基础配置
# layout:centered  wide
st.set_page_config(page_title='机器学习项目实战',
                   page_icon='🤖',
                   layout='centered',
                   )

st.title("🤖机器学习项目实战")
st.markdown('### 课题名称：租房价格预测')
st.markdown('**小组**：第08组')
st.divider()

st.markdown("#### 请选择预测条件：")
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
attribute, lng_lat, q_xq, cq = load_attribute.construct_attribute(file_name=os.path.join(ROOT_DIR, "dataset", "中国租房信息数据集.csv"))

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
    '最近医院距离': 3755,  # [-1,401]
    '周边医院个数': 2
}


def data_transfor(val):
    """
        数据转换：是-1，否-0
    """
    if val == '是':
        return 1
    else:
        return 0


@st.cache_data(show_spinner=True)
def update_lng_lat(xiao_qu=input_data['小区'], type=None):
    """
        经纬度获取
    """
    # print(xiao_qu)
    if 'lat' == type:
        # print(lng_lat[xiao_qu]['lat'])
        return lng_lat[xiao_qu]['lat']
    if 'lng' == type:
        # print(lng_lat[xiao_qu]['lng'])
        return lng_lat[xiao_qu]['lng']


@st.cache_data(show_spinner=True)
def getXQByQ(q=input_data['区']):
    """
        通过区获取小区
    """
    return q_xq[q]


@st.cache_data(show_spinner=True)
def getQByC(c=input_data['城市']):
    """
        通过城市获取区
    """
    return cq[c]


xq = None
q = None
c = None
st.write("**地域基本信息**")
input_data['租房网站名称'] = st.selectbox('租房网站名称', attribute['租房网站名称'])

c = st.selectbox('城市', attribute['城市'])
input_data['城市'] = c
q = st.selectbox('区', getQByC(c=c))
input_data['区'] = q
col11, col21, col31 = st.columns(3)

with col11:
    xq = st.selectbox('小区', getXQByQ(q=q))
    input_data['小区'] = xq
    if "lat" not in st.session_state:
        st.session_state.lat = update_lng_lat(xq, 'lat')
    else:
        st.session_state.lat = update_lng_lat(xq, 'lat')
    if "lng" not in st.session_state:
        st.session_state.lng = update_lng_lat(xq, 'lng')
    else:
        st.session_state.lng = update_lng_lat(xq, 'lng')
with col21:
    st.text_input(label='经度', value=st.session_state.lng, disabled=True)
    input_data['lng'] = st.session_state.lng
with col31:
    input_data['lat'] = st.session_state.lat
    st.text_input(label='纬度', value=st.session_state.lat, disabled=True)
st.write("**房间基本信息**")
# 房间基本信息
input_data['总楼层'] = st.number_input('总楼层', min_value=0, max_value=100, value=1)
input_data['所属楼层'] = st.number_input('所属楼层', min_value=0, max_value=100, value=1)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    input_data['室'] = st.number_input('室', min_value=1, max_value=10, value=1)
with col2:
    input_data['卫'] = st.number_input('卫', min_value=0, max_value=10, value=0)
with col3:
    input_data['厅'] = st.number_input('厅', min_value=0, max_value=10, value=0)
with col4:
    input_data['面积'] = st.number_input("房间面积:", min_value=0.0, max_value=10000.0, value=10.0, step=0.1)
with col5:
    input_data['朝向'] = st.selectbox('朝向', ['南', '东西', '西南', '东', '西', '东北', '西北', '北'])

st.write("**房间内基础配套信息**")
oth1, oth2, oth3, oth4 = st.columns(4)
with oth1:
    input_data['是否有阳台'] = data_transfor(st.radio('是否有阳台', ['是', '否'], horizontal=True))

with oth2:
    input_data['是否有床'] = data_transfor(st.radio('是否有床', ['是', '否'], horizontal=True))
with oth3:
    input_data['是否有衣柜'] = data_transfor(st.radio('是否有衣柜', ['是', '否'], horizontal=True))
with oth4:
    input_data['是否有沙发'] = data_transfor(st.radio('是否有沙发', ['是', '否'], horizontal=True))

oth5, oth6, oth7, oth8 = st.columns(4)
with oth5:
    input_data['是否有电视'] = data_transfor(st.radio('是否有电视', ['是', '否'], horizontal=True))
with oth6:
    input_data['是否有冰箱'] = data_transfor(st.radio('是否有冰箱', ['是', '否'], horizontal=True))

with oth7:
    input_data['是否有洗衣机'] = data_transfor(st.radio('是否有洗衣机', ['是', '否'], horizontal=True))
with oth8:
    input_data['是否有空调'] = data_transfor(st.radio('是否有空调', ['是', '否'], horizontal=True))

oth9, oth10, oth11, oth12 = st.columns(4)
with oth9:
    input_data['是否有热水器'] = data_transfor(st.radio('是否有热水器', ['是', '否'], horizontal=True))
with oth10:
    input_data['是否有宽带'] = data_transfor(st.radio('是否有宽带', ['是', '否'], horizontal=True))
with oth11:
    input_data['是否有燃气'] = data_transfor(st.radio('是否有燃气', ['是', '否'], horizontal=True))
with oth12:
    input_data['是否有暖气'] = data_transfor(st.radio('是否有暖气', ['是', '否'], horizontal=True))

st.write("**周边环境信息**")
col5, col6, col7, col8 = st.columns(4)
# 最近学校距离,周边学校个数,最近医院距离,周边医院个数
with col5:
    input_data['周边学校个数'] = st.number_input('周边学校个数', min_value=0, max_value=100, value=0)
with col6:
    input_data['最近学校距离'] = st.number_input('最近学校距离(单位：米)', min_value=0, max_value=3000, value=0)
with col7:
    input_data['周边医院个数'] = st.number_input('周边医院个数', min_value=0, max_value=100, value=0)
with col8:
    input_data['最近医院距离'] = st.number_input('最近医院距离(单位：米)', min_value=0, max_value=5000, value=0)

submitted = st.button('模型预测')

# # 模型加载
model_path = os.path.join(ROOT_DIR, "page_code", "model", "adaboost.joblib")
model = load_model.model_load(model_path)

input_df = pd.DataFrame([input_data])
# # 模型预测 
y_pred = None
if submitted:
    y_pred = model.predict(input_df)
    # print(type(y_pred))
    if len(y_pred) > 0:
        # total price
        total_price = y_pred[0] * input_data['面积']
        print(f'Predicted price of one month: {int(total_price)}￥')
        st.markdown(f"**月租金预测结果为：**：{int(total_price)}￥")
    else:
        st.markdown(f"**月租金预测结果为：**：预测失败")

# input_data
