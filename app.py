import streamlit as st
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_unstructured import UnstructuredLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from utils.conversational_chain import LLMHandler
from utils.summary_chain import SummaryDocument
import config.chain_config as cfg
import os
import shutil

# Initialize the LLM
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name=cfg.model_name, temperature=cfg.temperature)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# llm = Ollama(model= cfg.model_name)
# embeddings = FastEmbedEmbeddings()

FILE_TYPES = ['txt', 'pdf', 'docx', 'doc', 'csv']

# Ensure that all necessary directories exist
if not os.path.exists("temp"):
    os.makedirs("temp")

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    
    shutil.rmtree(cfg.VECTOR_STORE_DIR, ignore_errors=True)

# Streamlit UI components
st.title("Smart RAG ChatBot")
st.sidebar.title("Upload Documents")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=FILE_TYPES, on_change=clear_cache)

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = f"temp/{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize the document and conversation handler
    conversation_handler = LLMHandler(
        llm, 
        temp_file_path, 
        embeddings
    )

    st.sidebar.success("Document uploaded successfully!")

    # Chat with the document
    st.header("RAG Your Document")
    user_input = st.text_input("Ask a question about the document:")
    
    if user_input:
        with st.spinner("Generating response..."):
            conversation_handler.chat(user_input)
            st.subheader("Chat History")
            for message in conversation_handler.chat_history:
                if isinstance(message, HumanMessage):
                    st.markdown(f"**You:** {message.content}")
                elif isinstance(message, AIMessage):
                    st.markdown(f"**RAGBot:** {message.content}")

else:
    st.sidebar.info("Please upload a document to get started.")
