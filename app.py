import os
import fitz
import mimetypes
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def get_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')
    return vector_store

def get_conversational_chain(vector_store):
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-001')
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    conversational_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return conversational_chain

def process_chat(conversational_chain, user_prompt, chat_history):
    response = conversational_chain.invoke({
        'input': user_prompt,
        'chat_history': chat_history
    })
    return response['answer']

def main():
    load_dotenv()
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

    st.set_page_config(page_title='DocuQuery', page_icon='🤖', layout='wide')
    st.sidebar.title('TalkieAI')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'conversational_chain' not in st.session_state:
        st.session_state.conversational_chain = None

    user_prompt = st.chat_input("Your message here...")
    uploaded_file = st.sidebar.file_uploader('Upload Files', type=['pdf'])
    if uploaded_file is not None:
        mime_type = mimetypes.guess_type(uploaded_file.name)[0]
        if mime_type == 'application/pdf':
            text = get_text_from_pdf(uploaded_file)
            vector_store = get_vector_store(text)
            st.session_state.conversational_chain = get_conversational_chain(vector_store)
            st.sidebar.success("PDF processed successfully.")
        else:
            st.sidebar.write("Unsupported file type.")
    
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    if user_prompt and st.session_state.conversational_chain:
            response = process_chat(st.session_state.conversational_chain, user_prompt, st.session_state.chat_history)
            assistant_message = f"{response}"

            st.session_state.chat_history.append(HumanMessage(content=user_prompt))
            st.session_state.chat_history.append(AIMessage(content=response))

            with st.chat_message("user"):
                st.write(user_prompt)

            with st.chat_message("assistant"):
                st.write(assistant_message)
    
if __name__ == '__main__':
    main()