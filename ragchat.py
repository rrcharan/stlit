__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
import openai
import datetime
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
# Self query retriever
from langchain_community.llms import OpenAI
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.chains.query_constructor.base import AttributeInfo
# Compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# 
from langchain_openai  import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Document loader

loaders = [PyPDFLoader('Exam-Guide.pdf')]

docs= []
for loader in loaders:
    docs.extend(loader.load())

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=150,
    # length_function=len,
    )
 
splits = text_splitter.split_documents(docs)

# Embeddings
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Vector store
persist_directory = os.path.join(os.getcwd(), 'docschat/chroma/')

# new db creation
vectordb = Chroma.from_documents(documents=splits, 
                                 embedding=embedding, 
                                 persist_directory=persist_directory)

vectordb.persist()

# model name 
llm_name = "gpt-3.5-turbo"

# build prompt
def gen_prompt(context, user_text):

    llm = ChatOpenAI(model_name= llm_name, temperature=0, openai_api_key=openai_api_key)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={"prompt" : QA_CHAIN_PROMPT}
    )

    result = qa_chain({"query": user_text})

    st.info(result["result"])
    result["source_documents"]

with st.form('my_form'):
   
    user_text = st.text_area('Enter User Question:', '')
    context = ''
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')

    if submitted:
        gen_prompt(context, user_text)
