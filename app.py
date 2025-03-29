
import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

from myrag import MyRAG

load_dotenv()

APP_TITLE=os.getenv('MYRAG_APP_TITLE', 'MyRAG')
APP_SUBTITLE=os.getenv('MYRAG_APP_SUBTITLE', 'Ask me anything!')

st.set_page_config(page_title=APP_TITLE)

def display_messages():
    st.subheader(APP_SUBTITLE)
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i), allow_html=True)
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            _, data = st.session_state["assistant"].query(user_text)

        agent_text = f"{data['response']}<small style='color: #787878;'>\n\nSources:\n{data['sources']}\n\nContext:\n{data['context']}</small?"
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = MyRAG()

    st.header("🛰 "+APP_TITLE)
    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
