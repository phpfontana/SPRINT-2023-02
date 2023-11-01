import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def load_documents():
    """
    Load pdf files from directory and split text into chunks
    """
    # Verify if the data directory exists
    if not os.path.isdir("src/data/"):
        st.error("The src/data directory does not exist. Please create it and add some pdf files.")
        st.stop()

    # Verify if there are pdf files in the data directory
    if len(os.listdir("src/data/")) == 0:
        st.error("There are no pdf files in the data directory. Please add some pdf files.")
        st.stop()

    # Load pdf files from directory and split text into chunks
    loader = DirectoryLoader('src/data/', glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def split_text_into_chunks(documents):
    """
    Split text into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})


def create_vector_store(text_chunks, embeddings):
    return FAISS.from_documents(text_chunks, embeddings)

