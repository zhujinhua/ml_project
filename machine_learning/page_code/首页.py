import streamlit as st

# 页面基础配置
st.set_page_config(page_title='机器学习项目实战', 
                   page_icon='🤖', 
                   layout='wide',
                   )

# print(st.__version__)
# 标题
st.title("🤖机器学习项目实战")
st.markdown('### 课题名称：租房价格预测')
st.markdown('**数据来源**：http://www.idatascience.cn/dataset-detail?table_id=100086')
st.markdown('**小组**：第08组')
# st.markdown('姓名：王二狗')
st.markdown("**小组PM**：朱金华")
st.markdown("**小组成员**：陈天、何韵、林参元、王星、舒小芳、蔡娟、王亮、桂龙、陈轩、葛鹏、刘亮、王艳、杨小刚、康娅、陈永辉、王敬阳、周毅、孙仕娟、陈而芳、陈俊材")

# 分割线
# st.divider()
# 侧边栏
# st.sidebar