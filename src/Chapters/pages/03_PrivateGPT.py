from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import GPT4All

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
import streamlit as st

# 이 섹션에서는 document GPT를 private GPT로 전환할 것이다.
# OpenAI를 사용하지 않고 LLM모델을 임베딩할 것이다. 인터넷을 전혀 사용하지 않음.

#방법 종류 Hugging Face, GPT4All, Ollama
st.set_option("CUDA_VISIBLE_DEVICES", "0")
st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📓"
)

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
            

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# llm = ChatOpenAI(
#     temperature=0.1, 
#     streaming=True, 
#     callbacks=[
#         ChatCallbackHandler()
#     ]
# )
llm = CTransformers(
    model="./files/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    temperature=0.1, 
    max_new_tokens=256,
    streaming=False, 
    callbacks=[
        ChatCallbackHandler()
    ]
)


# 메모리 객체 정의.
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=256,
        memory_key="chat_history",
        return_messages=True,
    )
memory = st.session_state['memory'] 

def get_history(_):
    return memory.load_memory_variables({})

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding files...")   # 매번 실행되지 않게 캐시 확인.
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir 
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
    

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)



def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(_):
    return memory.load_memory_variables({}).get("chat_history", [])

st.title("PrivateGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        You are a professional chef. Please give me the recipe for the food I am asking about reply in korean.
        """
     ),
    ("human", "<s>[INST]{question}</s>")
])
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system", 
#         """
#         Answer the question using ONLY the following context. If you don't know the answer
#         just say that you don't know. DO NOT give any explanations.

#         Context: {context}
#         """
#      ),
#      MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{question}")
# ])


def get_history():
    return memory.load_memory_variables({})


with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf .docx file", type=["pdf","txt","docx", "md"])

if file:
    # retriever = embed_file(file)
    
    send_message("준비 완료!! 물어봐!!!", "ai", save=False)
    paint_history()
    message = st.chat_input("파일에 관해 뭐든지 물어봐!")

    if message:
        send_message(message, "human")
        chain = {
          
            "question": RunnablePassthrough(),
        }|RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm 
        # chain = {
        #     "context": retriever | RunnableLambda(format_docs),
        #     "question": RunnablePassthrough(),
        # } |RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm 
       
        with st.chat_message("ai"):
            response = chain.invoke(message)
            # print(response)
            memory.save_context({"input": message}, {"output": response })
else:
    st.session_state["messages"] = [] 
