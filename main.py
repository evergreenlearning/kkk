import streamlit as st
import os
from langchain_ollama.chat_models import ChatOllama
from langchain_community.chat_models import MoonshotChat, ChatTongyi
from langchain.memory import ConversationBufferMemory
import base64

# ========== 头像处理 ==========
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 加载头像图片
user_avatar_path = "human.png"
ai_avatar_path = "bot.png"
user_avatar_base64 = img_to_base64(user_avatar_path)
ai_avatar_base64 = img_to_base64(ai_avatar_path)

# ========== 初始化对话记忆 ==========
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 获取模型流式响应 ==========
def get_chat_response_stream(prompt, model, memory):
    llm = MoonshotChat(api_key="")
    # llm = ChatOllama(model=model)
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
        avatar = ai_avatar_base64
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
st.title("KK-AI助手")

# 选择模型
models = [
    "qwen2:7b",
    "qwen2.5:latest",
    "deepseek-r1:latest"
]
select_model = st.sidebar.selectbox(label="选择模型", options=models)

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

        # 记录 AI 回复
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.memory.save_context({"input": prompt}, {"output": response})

        # 再次渲染 AI 消息（带头像）
        message_placeholder.empty()  # 清除占位符
        display_message("assistant", response)
