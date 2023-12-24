import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today)
# st.write(p)
model = st.selectbox("선택하세요", ["GPT-3", "GPT-4"])

if model == "GPT-3":
    st.write("싸다")
else:
    st.write("비싸다")

    st.write(model)

    name = st.text_input("이름을 입력하세요")

    st.write(name)

    value = st.slider("temperature", min_value=0.1, max_value=1.0, )

    st.write(value)