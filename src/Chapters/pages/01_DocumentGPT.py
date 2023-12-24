import streamlit as st
import time


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📓"
)

st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
        if save:
            st.session_state["messages"].append({"message": message, "role": role})

# st.write(st.session_state["messages"])
for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], False)


message = st.chat_input("AI에게 메시지를 보내세요!")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)