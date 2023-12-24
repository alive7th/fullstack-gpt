import streamlit as st
from langchain.prompts import PromptTemplate


st.title("Hello World!")

st.subheader("Welcome to Streamlit!")

st.markdown("""
    #### I love it!!
""")

st.write("Hellow Im Jungwon")

# st.write([1,2,3,4,5])
a = [1,2,3,4,5]
a

# st.write({"name": "jwkim"})
b = {"name": "jwkim"}

b

# st.write(PromptTemplate)
PromptTemplate

p = PromptTemplate.from_template("sssss")
p
# st.write(p)
st.selectbox("선택하세요", ["GPT-3", "GPT-4"])