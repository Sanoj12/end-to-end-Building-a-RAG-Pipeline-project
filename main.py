import faiss
import numpy as np
import sys
import streamlit as st
from app import get_wikipedia_data,split_text,embeddings,embedding_model,index

st.title("Q&A RAG APPLICATION 🚀")
st.write("Hello, this is a Q&A application based on wikipedia content")

# Input and output
user_input = st.text_input("enter the topic:")

if st.button("submit & process"):
    with st.spinner("proccessing your wikipedia data..."):
     get_data=get_wikipedia_data(user_input)
     text_chunks =split_text(get_data)
     embeddings = embedding_model.encode(text_chunks)
     dimension =embeddings.shape[1] 
     index= faiss.IndexFlatL2(dimension)

     index.add(np.array(embeddings))

if user_input:
    st.write(f"Please ask the question about the topic : {user_input}")
    question = st.text_input("ask the question:")


