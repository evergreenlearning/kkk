from langchain.chains import ConversationChain
from langchain_community.chat_models import MoonshotChat, ChatTongyi
from langchain_ollama.chat_models import ChatOllama
import os
os.environ["MOONSHOT_API_KEY"] = "sk-T009onh3gF4BQonapjO6KFV6CDxsV8XIMaPP1sj13eNTw0oe"
def get_chat_response(prompt, model, memory):
    # llm = ChatOllama(model=model)
    llm = MoonshotChat()
    chain=ConversationChain(llm=llm, memory=memory)
    response=chain.invoke({"input":prompt})
    return response["response"]

