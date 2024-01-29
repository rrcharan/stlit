import streamlit as st
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import  SystemMessage, HumanMessage
from langchain.prompts import HumanMessagePromptTemplate

st.title('🦜🔗 Marketing message generation App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# product_items = st.sidebar.text_input('Items')

#  call to LLM model

def generate_response(sys_text, user_text):
    st.write(sys_text, user_text)
    template = ChatPromptTemplate.from_messages([
        SystemMessage(content=(f"{sys_text}")),
        # Human message
        (HumanMessagePromptTemplate.from_template("{user_text}") ),

    ])

    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    messages = template.format_messages(user_text=user_text, sys_text=sys_text)
    st.write(messages)
    response = llm.invoke(messages)
    # response = llm(template.format_messages(user_text=user_text, system_message=sys_text))
    print(response.content)
    st.info(response.content)
# Finally, use st.form() to create a text box (st.text_area()) for user input. When the user clicks Submit, 
# the generate-response() function is called with the user's input as an argument.
    
with st.form('my_form'):
    sys_text = st.text_area('Enter System instructions:', '')    
    user_text = st.text_area('Enter User Prompt:', '')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if not sys_text and not user_text:
        st.warning('Please enter system and user text!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(sys_text, user_text)