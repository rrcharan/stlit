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
from langchain_community.vectorstores import Chroma
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import hub
import getpass
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage

# langsmith
from langsmith.run_trees import RunTree

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY"

print(" start of the program/n")
# Document loader



# Text splitter


# Embeddings
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Vector store
persist_directory = os.path.join(os.getcwd(), 'docschat/chroma/')

# new db creation
vectordb = Chroma(embedding_function=embedding, 
                                 persist_directory=persist_directory)

# model name 
llm_name = "gpt-3.5-turbo"

# memory
# memory = ConversationBufferMemory(
#     memory_key='chat_history',
#     return_messages=True,
#     )
# retriever
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)   

print("running chat init")
chat_history = []

# callback to save state session
if 'history' not in st.session_state:
        st.session_state['history'] = []


# build prompt
def gen_prompt(context, user_text, chat_history):

    llm = ChatOpenAI(model_name= llm_name, temperature=0, openai_api_key=openai_api_key)

    print("\nchat history in method \n:---", chat_history)
    print("user_text:---", user_text)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    
    # qa_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | QA_CHAIN_PROMPT
    #     | llm
    #     | StrOutputParser()
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )    
    


    result = qa({"question": user_text, "context": context, "chat_history": chat_history})
    
   
    docs_ref = result["source_documents"]
   
    print("\n-----------------------------------\n")
    print("docs reference:", docs_ref)
    st.title('Response from LLM')
    st.info(result['answer'])
    st.title('Question Generated')
    st.info(result['generated_question'])
    return result
    # print(result)
    

container = st.container()

tab1, tab2 = st.tabs(["Q&A", "History"])

with tab1:
    with container:

        with st.form('my_form'):
        
            user_text = st.text_area('Enter User Question:', '')
            context = ''
            submitted = st.form_submit_button('Submit')
            chat_clear = st.form_submit_button('Chat_Clear', on_click=lambda: st.session_state['history'].clear())
            if not openai_api_key.startswith('sk-'):
                st.warning('Please enter your OpenAI API key!', icon='⚠')

            if submitted:
                print("\n----before submission--------------------------\n")
                print("chat history:---", chat_history)
                response = gen_prompt(context, user_text, chat_history=st.session_state['history'])
                st.session_state['history'].extend([(user_text, response["answer"])])
                chat_history = st.session_state['history']

                # st.session_state['query'].append(user_text)
                # st.session_state['generated'].append(response["answer"])
                print("\n-----------------------------------\n")
                print("chat history:---", chat_history)


with tab2:
     
    st.markdown("Chat History")
  

    # display chat history
    hist_container = st.container()

    if st.session_state['history']:
        with hist_container:
            for i, (query, response) in enumerate(st.session_state['history']):
                st.write(f"Query: {query}")
                st.write(f"Response: {response}")
                st.write("----")