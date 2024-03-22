import streamlit as st
import time

import os

from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        # assign role based on chat history order
        if i % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        # display chat messages from history on app rerun
        with st.chat_message(role):
            st.markdown(message.content)

def create_conversation(prompt):
    # get response from llm
    response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = response['chat_history']

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        st.write_stream(response_generator(response['answer']))


def response_generator(answer):
    for word in answer.split(' '):
        yield word + " "
        time.sleep(0.05)

def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None


def main():
    st.set_page_config(page_title="LLM With Evidence", 
                    page_icon=":nerd_face:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("LLM With Evidence ‒ Chat with PDFs :books:")

    if st.session_state.chat_history is not None:
        display_chat_history()

    if prompt := st.chat_input("Ask a question about your documents!"):
        create_conversation(prompt)
        

        

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
                "Upload your PDFs here and press on 'Process' button", accept_multiple_files=True)
        if st.button("Process", type="primary"): # If the "Process" button is pressed, ...
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # divide into chunks
                text_chunks =  get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                st.markdown("Successfully loaded documents!")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        # Reset chat button
        if st.button('Reset Chat', on_click=reset_conversation):
            st.markdown("The conversation has been reset.")



if __name__ == '__main__':
    main()