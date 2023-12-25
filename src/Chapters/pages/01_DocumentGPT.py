import streamlit as st
import time
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📓"
)

def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.docx")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir 
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver

st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader("Upload a .txt .pdf .docx file", type=["pdf","txt","docx"])

if file:
    retriever = embed_file(file)
    retriever.invoke("개울가") 

# for doc in list of docs | prompt | llm

# for response in list of llms response | put them all together
# map_doc_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
#             Use the following portion of a long document to see if any of the text is relevant to
#             answer the question. Return any relevant text verbatim.
#             ------
#             {context}
#         """
#     ),
#     (
#         "human", "{question}"
#     )
# ])