from langchain.chains import ConversationChain
from langchain_community.chat_models import MoonshotChat, ChatTongyi
from langchain_ollama.chat_models import ChatOllama
def get_chat_response(prompt, model, memory):
    llm = ChatOllama(model=model)
    chain=ConversationChain(llm=llm, memory=memory)
    response=chain.invoke({"input":prompt})
    return response["response"]

