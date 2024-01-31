import os
import shutil
import openai
import sys
import datetime
import streamlit as st
import sqlite3
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# Self query retriever
from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
# Compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# 
from langchain_openai  import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


st.title('🦜🔗 RAG DEMO')

# _ = load_dotenv(find_dotenv())

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# open_api_key = os.environ['OPENAI_API_KEY']
# loader = PyPDFLoader('ML.pdf')
# uncomment to create new embeddings **************************************
# loaders = [PyPDFLoader('Exam-Guide.pdf'),
#         #    PyPDFLoader('Sample-Questions.pdf'),
#            ]
# ---------------------------------------------------------------------------
# single document
# pages = loaders.load()
#  each page is a document
# print(len(pages))
# page = pages[0]
# print(page)
# print(page.page_content[0:100])
# page.metadata
# uncomment to create new embeddings **************************************
# for multiple documents, load one by one
# docs = []
# for loader in loaders:
#     docs.extend(loader.load())
# -----------------------------------  ------------------------------------
# RecursiveCharacterTextSplitter is recommended for generic text.
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1000,
#     chunk_overlap=150,
#     length_function=len,
#         )

# docs = text_splitter.split_documents(pages)
# print(len(docs))
# print(len(pages))
# Uncomment to create new embeddings **************************************
# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", " ", ""],
#     chunk_size=1000,
#     chunk_overlap=150,
#     # length_function=len,
#     )

# splits = text_splitter.split_documents(docs)
# print(len(splits))

# # Embeddings
embedding = OpenAIEmbeddings()

# # vector stores
persist_directory = os.path.join(os.getcwd(), 'docs2/chroma/')

# to query existing embeddings, change function to Chroma, remove from_documents
# vectordb = Chroma(
#     # when new db creted, use splits?
#     documents=splits,
#     # to query existing embeddings, change paramter embedding to embedding_function
#     embedding=embedding,
#     # embedding_function=embedding,
#     persist_directory=persist_directory)
# new db creation
# vectordb = Chroma.from_documents(documents=splits, 
#                                  embedding=embedding, 
#                                  persist_directory=persist_directory)

# print(vectordb._collection.count())
# Existing db query
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory)
# ---------------------------------------------------------------------
# sample similarity search
# question = "what are Storage services?"
# docs = vectordb.similarity_search(question, k=5)
# print(len(docs))
# print(docs[0].page_content)
# !rm -rf ./docs/chroma  # remove old database files if any
# if os.path.exists(persist_directory):
#     shutil.rmtree(persist_directory)
#     print(doc.metadata)
# vectordb.persist()

# Maximum marginal relevance-----------------------------------------------
# self-query retriever - uses LLM to extract the query string to use for vector search and a meta data filter to pass in as well. (user query converted to search query)
#  Sample code --------------------------------
# metadata_field_info = [
#     AttributeInfo(
#         name="source",
#         description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
#         type="string",
#     ),
#     AttributeInfo(
#         name="page",
#         description="The page from the lecture",
#         type="integer",
#     ),
# ]

# document_content_description = "Lecture Notes"
# llm = OpenAI(temperature=0)
    

# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectordb,
#     document_content_description,
#     metadata_field_info,
#     verbose=True,
# )

# question = "what did they say about regression in the third lecture?"

# docs = retriever.get_relevant_documents(question)

# for d in docs:
#     print(d.metadata)
#  Sample code --------------------------------
# Compression
# def pretty_print_docs(docs):
#     print(f"\n {'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Wrap our vectorstore------------------    
# compressor = LLMChainExtractor.from_llm(llm)

# compressor_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=vectordb.as_retriever()
# )

# question = "what did they say about matlab?"
# compressed_docs = compressor_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)

# Combining various techniques-----------------------------------------------
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vectordb.as_retriever(search_type = "mmr")
# )

# question = "what did they say about matlab?"
# compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)
# -----------------------------------------------   
current_date = datetime.datetime.now().date()

if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
# print(llm_name)

llm = ChatOpenAI(model_name= llm_name, temperature=0, openai_api_key=openai_api_key)

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
# )

# result = qa_chain({"query":question})
# result['result']

# ---------------------------------------------------------------------------

# build prompt


def generate_prompt(context, user_text):

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Run chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={"prompt" : QA_CHAIN_PROMPT}
    )

    # question = "what are Storage services?"
    # question = user_text
    # print('queryin.....')
    result = qa_chain({"query": user_text})
    # print('waiting for repsonse.....')
    # print(result["result"])
    st.info(result["result"])
    result["source_documents"]

with st.form('my_form'):
    # sys_text = st.text_area('Enter System instructions:', '')    
    user_text = st.text_area('Enter User Question:', '')
    context = ''
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    # if not sys_text and not user_text:
    #     st.warning('Please enter system and user text!', icon='⚠')
    if submitted:
        generate_prompt(context, user_text)