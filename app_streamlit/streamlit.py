import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)

import streamlit as st
from dotenv import load_dotenv
# import app_streamlit.retrieve as llama2
from app_streamlit.retrieve import build_chain, run_chain

load_dotenv()

MAX_HISTORY_LENGTH = int(os.environ.get('MAX_HISTORY_LENGTH'))

# st.session_state["llm_app"] = llama2
# st.session_state["llm_chain"] = llama2.build_chain() # i.e. ConversationalRetrievalChain

st.set_page_config(page_title="AIAssistant-Aznor", page_icon="🧑‍💼")


def render_app():
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
        <style>
        button {
            height: 30px !important;
            width: 150px !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.subheader("Hello, I am an AI Assistant. \n\n I am here to share more about Aznor and how he can be an asset to GenAI CoE as a Generative AI Platform Engineer. \n\n Ask me anything in the chat box below.")

    # User input
    st.container()
    st.container()

    if "chat_dialogue" not in st.session_state:
        st.session_state["chat_dialogue"] = []

    # if "llm" not in st.session_state:
    #     st.session_state["llm"] = llama2
    #     st.session_state["llm_chain"] = llama2.build_chain()

    def clear_history():
        st.session_state["chat_dialogue"] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.chat_dialogue) >= MAX_HISTORY_LENGTH:
        clear_history()
    
    chain = build_chain()

    if prompt := st.chat_input("Type your questions here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display message from LLM
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer = ""
            # for dict_message in st.session_state.chat_dialogue:
            #     if dict_message["role"] == "user":
            #         string_dialogue = "User: " + dict_message["content"] + "\n\n"
            #     else:
            #         string_dialogue = "Assistant: " + dict_message["content"] + "\n\n"
            # llm_chain = st.session_state["llm_chain"]
            # chain = st.session_state["llm_app"]
            try:
                output = run_chain(chain, prompt, st.session_state.chat_dialogue)
            except Exception:
                output = {}
                output["answer"] = "I am sorry I am unable to respond to your question."
            answer = output.get("answer")
            if 'source_documents' in output:
                with st.expander("Sources"):
                    for _sd in output.get('source_documents'):
                        _sd_metadata = _sd.metadata
                        source = _sd_metadata.get('source')
                        title = _sd_metadata.get('title')
                        st.write(f"{title} --> {source}")
            answer_placeholder.markdown(answer + "▌")
            # Add user message to chat history
            st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
            # Add assistant response to chat history
            st.session_state.chat_dialogue.append({"role": "assistant", "content": answer})
        col1, col2 = st.columns([10, 4])
        with col1:
            pass
        with col2:
            st.button("Clear History", use_container_width=True, on_click=clear_history)

render_app()




