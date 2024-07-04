# Streamlit 应用程序界面
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_wenxin import Wenxin


def generate_response(input_text, wenxin_api_key, wenxin_app_secret):
    llm = Wenxin(
        temperature=0.8,
        model="ernie-speed-128k",
        baidu_api_key=wenxin_api_key,
        baidu_secret_key=wenxin_app_secret,
        verbose=True,
    )
    return st.info(llm(input_text))


def get_vectordb():
    persist_directory = '/Users/k1ra/Documents/经济学原理(第7版)'

    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb


def get_chat_qa_chain(question:str, wenxin_api_key:str, wenxin_app_secret:str):
    vectordb = get_vectordb()
    llm = Wenxin(
        temperature=0.8,
        model="ernie-speed-128k",
        baidu_api_key=wenxin_api_key,
        baidu_secret_key=wenxin_app_secret,
        verbose=True,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']


def get_qa_chain(question:str, wenxin_api_key:str, wenxin_app_secret:str):
    vectordb = get_vectordb()
    llm = Wenxin(
        temperature=0.8,
        model="ernie-speed-128k",
        baidu_api_key=wenxin_api_key,
        baidu_secret_key=wenxin_app_secret,
        verbose=True,
    )
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
            案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
            {context}
            问题: {question}
            """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


def main():
    st.title('🐽 Kira大模型应用开发')
    wenxin_api_key = '8O8u7ptN1XJkfdGXMvODRv1o'
    wenxin_app_secret = st.sidebar.text_input("App Secret", type="password")

    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    messages = st.container(height=300)
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if prompt := st.chat_input("写点吧"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

    if selected_method == "None":
        # 调用 respond 函数获取回答
        answer = generate_response(prompt, wenxin_api_key, wenxin_app_secret)
    elif selected_method == "qa_chain":
        answer = get_qa_chain(prompt, wenxin_api_key, wenxin_app_secret)
    elif selected_method == "chat_qa_chain":
        answer = get_chat_qa_chain(prompt, wenxin_api_key, wenxin_app_secret)

    # 检查回答是否为 None
    if answer is not None:
        # 将LLM的回答添加到对话历史中
        st.session_state.messages.append({"role": "assistant", "text": answer})

    # 显示整个对话历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            messages.chat_message("user").write(message["text"])
        elif message["role"] == "assistant":
            messages.chat_message("assistant").write(message["text"])


if __name__ == "__main__":
    main()
