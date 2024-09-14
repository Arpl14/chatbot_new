#!/usr/bin/env python
# coding: utf-8

# In[19]:

import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Keep only one embeddings import, either OpenAIEmbeddings or GoogleGenerativeAIEmbeddings based on what you're using.
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma  # For in-memory vector store
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google Generative AI embeddings and chat models
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
# Optionally, keep `genai` if you're using Google's API directly
import google.generativeai as genai
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma


# In[35]:


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Extract all of the texts from the PDF files 
def get_pdf_text(pdfs):
  text = ""
  for pdf in pdfs:
    pdf_reader = PdfReader(pdf)

    # iterate through all the pages in the pdf
    for page in pdf_reader.pages:
      text += page.extract_text()
    
    return text 

# Split text into chunks 
def generate_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=1000
  )

  chunks = text_splitter.split_text(text)

  return chunks 

# Convert Chunks into Vectors


# Modify the chunks_to_vectors function to use in-memory storage

# Modify the chunks_to_vectors function to avoid SQLite
def chunks_to_vectors(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Force Chroma to use DuckDB instead of SQLite, and avoid persistence on disk
    settings = Settings(
        chroma_db_impl="duckdb+parquet",  # Use DuckDB with Parquet, bypassing SQLite
        persist_directory=None,  # No persistence, completely in-memory
        anonymized_telemetry=False  # Disable telemetry
    )
    
    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name="document_embeddings", 
        embedding_function=embeddings,
        client_settings=settings  # Pass the DuckDB settings
    )
    
    # Add the chunks to the vector store
    vector_store.add_texts(chunks)
    
    return vector_store

def get_conversation():
    prompt_template = """
    Answer the question that is asked with as much detail as you can, given the context that has been provided. If you unable to come up with an answer based on the provided context,
    simply say "Answer cannot be provided based on the context that has been provided", instead of trying to forcibly provide an answer.\n\n
    Context: \n {context}?\n
    Question: \n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain 

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    chain = get_conversation()

    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])


# Main app portion of the project
# Main app portion of the project
def app():
    st.title("ASTM Documents Chatbot")
    st.sidebar.title("Upload Documents")

    # Sidebar
    pdf_docs = st.sidebar.file_uploader("Upload your documents in PDF format, then click Analyze.", accept_multiple_files=True)

    analyze_triggered = st.sidebar.button("Chat Now")

    if analyze_triggered:
        with st.spinner("Configuring... ⏳"):
            raw_text = get_pdf_text(pdf_docs)
            chunks = generate_chunks(raw_text)
            chunks_to_vectors(chunks)
            st.success("Done")

    # User Input 
    user_question = st.text_input("Ask a question based on the documents that were uploaded")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    app()





# In[ ]:





# In[ ]:





# In[ ]:




