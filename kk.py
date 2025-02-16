import streamlit as st
import os
from langchain_ollama.chat_models import ChatOllama
from langchain_community.chat_models import MoonshotChat, ChatTongyi
from langchain.memory import ConversationBufferMemory
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import base64

# ========== 环境变量 ==========
os.environ["MOONSHOT_API_KEY"] = "sk-T009onh3gF4BQonapjO6KFV6CDxsV8XIMaPP1sj13eNTw0oe"
os.environ["DASHSCOPE_API_KEY"] = "sk-987f7ca356da440cb996b059ab846303"

# ========== 头像处理 ==========
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
# 加载头像图片
user_avatar_path = "./common_data/human.png"
user_avatar_base64 = img_to_base64(user_avatar_path)


# ========== 初始化对话记忆 ==========
if "conversations" not in st.session_state:
    st.session_state.conversations = {"默认对话": []}
    st.session_state.current_conversation = "默认对话"
    st.session_state.memory = ConversationBufferMemory()

# ========== 选择或新建对话 ==========
st.sidebar.title("对话管理")
conversation_names = list(st.session_state.conversations.keys())
selected_conversation = st.sidebar.selectbox("选择对话", conversation_names)
new_conversation_name = st.sidebar.text_input("新建对话名称")
if st.sidebar.button("新建对话") and new_conversation_name:
    if new_conversation_name not in st.session_state.conversations:
        st.session_state.conversations[new_conversation_name] = []
        st.session_state.current_conversation = new_conversation_name
        st.session_state.memory = ConversationBufferMemory()
        st.rerun()

st.session_state.current_conversation = selected_conversation
st.session_state.messages = st.session_state.conversations[selected_conversation]

# ========== 获取模型流式响应 ==========
def get_chat_response_stream(prompt, model, memory):
    global llm
    
    if "deepseek" in model and "online" in model:
        deepseek_api_key = "sk-2148ec7d29cd4570ae91fdb15a81cc1d"
        nvidia_api_key = "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"
        if "r1" in model:
            llm = ChatNVIDIA(
                model="deepseek-ai/deepseek-r1",
                api_key=nvidia_api_key
            )
        elif "v3" in model:
            llm = ChatDeepSeek(model="deepseek-chat", api_key=deepseek_api_key)
    elif "qwen" in model:
        if "max" in model:
            llm = ChatTongyi(model="qwen-max")
        elif "plus" in model:
            llm = ChatTongyi(model="qwen-plus")
    elif model =="kimi-default-online":
        llm = MoonshotChat()
    elif "local" in model:
        llm = ChatOllama(model=model.split("-local")[0])
    history_data = memory.load_memory_variables({})
    chat_history = history_data.get("history", [])

    if not isinstance(chat_history, list):
        chat_history = []

    messages = chat_history + [{"role": "user", "content": prompt}]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content

# ========== 自定义消息显示（带头像） ==========
def display_message(role, content):
    """使用 HTML+CSS 定制化消息格式"""
    if role == "user":
        alignment = "flex-end"  # 右对齐
        avatar = user_avatar_base64
        color = "#f0f0f2"  # 灰色气泡
        st.markdown(
            f"""
                <div style="display: flex; justify-content: {alignment}; align-items: center; margin: 10px 0;">
                    <div style="background-color: {color}; padding: 10px; border-radius: 10px; max-width: 80%; word-wrap: break-word; font-size: 16px;">
                        {content}
                    </div>
                    <img src="data:image/png;base64,{avatar}" style="width: 35px; height: 35px; border-radius: 50%; margin-left: 10px;">
                </div>
                """,
            unsafe_allow_html=True
        )
    else:
        alignment = "flex-start"  # 左对齐
        color = "#f0f0f2"  # 灰色气泡
        st.markdown(
            f"""
            <div style="display: flex; justify-content: {alignment}; align-items: center; margin: 10px 0;">
                <div style="background-color: {color}; padding: 10px; border-radius: 10px; max-width: 80%; word-wrap: break-word; font-size: 16px;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ========== Streamlit 界面 ==========
title = "KK-AI助手"
st.markdown(f"""
    <h1 style="color: blue; font-size: 30px; font-weight: bold; text-align: center;">{title}</h1>
    """, unsafe_allow_html=True)

# 选择模型
models = [
    "deepseek-r1:latest-local",
    "qwen-max-online",
    "qwen-plus-online",
    "deepseek-v3-online",
    "deepseek-r1-online",
    "kimi-default-online"
]
select_model = st.sidebar.selectbox(label="选择模型", options=models)
st.write(select_model)
# 显示历史对话（带头像）
for message in st.session_state.messages:
    display_message(message["role"], message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)

    # AI 回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = ""

        # 获取流式响应
        for chunk in get_chat_response_stream(prompt, model=select_model, memory=st.session_state.memory):
            response += chunk
            message_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.memory.save_context({"input": prompt}, {"output": response})
        st.session_state.conversations[selected_conversation] = st.session_state.messages  # 更新对话历史

        message_placeholder.empty()  # 清除占位符
        display_message("assistant", response)