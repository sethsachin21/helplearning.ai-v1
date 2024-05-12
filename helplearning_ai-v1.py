#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:42:06 2024

@author: sachin-kumar
"""


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

# Function to log question and answer to S3 or append to screen
def log_question_and_answer(question, answer):
    # Initialize responses if not present in session state
    if 'responses' not in st.session_state:
        st.session_state.responses = []

    # Count words in answer
    word_count = count_words(answer)

    # Generate serial number for the question
    serial_number = len(st.session_state.responses) + 1

    # Append new question, answer, and word count
    new_entry = f"**{serial_number}. Question:** {question}\n\n**Answer:** {answer}\n\n(Number of words: {word_count})"
    st.session_state.responses.insert(0, new_entry)

    # Display all responses
    for response in st.session_state.responses:
        st.markdown(response)


# Function to count words in text
def count_words(text):
    words = text.split()
    return len(words)

def process_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"An error occurred while processing the PDF file: {e}")
        return None


OPENAI_API_KEY = st.secrets['llm_api_key']

# Upload PDF files
st.header("helplearning.ai")

with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader("Upload PDF files and start asking questions", accept_multiple_files=True, type="pdf")

# Extract the text
if files:
        texts = [process_pdf(file) for file in files if file is not None]
        texts = [text for text in texts if text is not None]  # Remove None values

        if texts:
            try:
                # Concatenate all texts into a single string
                all_text = "\n".join(texts)

                # Break it into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators="\n",
                    chunk_size=200,
                    chunk_overlap=50,
                    length_function=len
                )
                chunks = text_splitter.split_text(all_text)

                # Generating embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

                # Creating vector store - FAISS
                vector_store = FAISS.from_texts(chunks, embeddings)

                # Get user question
                user_question = st.text_input("Type Your question here")

                # Do similarity search
                if user_question:
                    match = vector_store.similarity_search(user_question)

                    # Define the LLM
                    llm = ChatOpenAI(
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0,
                        max_tokens=500,
                        model_name="gpt-3.5-turbo"
                    )

                    # Output results
                    # Chain -> take the question, get relevant document, pass it to the LLM, generate the output
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=match, question=user_question)
                    log_question_and_answer(user_question, response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
