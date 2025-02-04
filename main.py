import streamlit as st
from utils import get_chat_response
from langchain.memory import ConversationBufferMemory
import base64

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 加载头像图片并转换为Base64
user_avatar_path = 'human.png'  # 替换为您的用户头像路径
ai_avatar_path = 'bot.png'  # 替换为您的AI头像路径
user_avatar_base64 = img_to_base64(user_avatar_path)
ai_avatar_base64 = img_to_base64(ai_avatar_path)

st.title("KK-AI助手")

# 选择模型
models = [
    "qwen2:7b",
    "qwen2.5:latest"
]
select_model = st.sidebar.selectbox(label="选择模型", options=models)
st.write(select_model)

# 初始化会话状态
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
    st.session_state["messages"] = [{"role": "ai", "content": "你好，我是你的AI助手，有什么可以帮你的吗？"}]


def display_message(message, user_avatar_base64, ai_avatar_base64):
    if message["role"] == "human":
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end; color: blue;"><div class="user-message">{message["content"]} <img class="avatar" src="data:image/png;base64,{user_avatar_base64}" style="width:25px;height:25px;" /></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-message" style="text-align: left; color: black;"><img class="avatar" src="data:image/png;base64,{ai_avatar_base64}" style="width:30px;height:30px;" /> {message["content"]}</div>',
            unsafe_allow_html=True,
        )

# 显示所有历史消息
for message in st.session_state["messages"]:
    display_message(message, user_avatar_base64, ai_avatar_base64)

# 用户输入
prompt = st.chat_input("在这里输入...")
if prompt:
    st.session_state["messages"].append({"role": "human", "content": prompt})
    display_message({"role": "human", "content": prompt}, user_avatar_base64, ai_avatar_base64)

    with st.spinner("AI正在飞速加载中..."):
        response = get_chat_response(prompt=prompt, model=select_model, memory=st.session_state["memory"])

    msg = {"role": "ai", "content": response}
    st.session_state["messages"].append(msg)
    display_message(msg, user_avatar_base64, ai_avatar_base64)