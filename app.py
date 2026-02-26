import streamlit as st
import os
from rag_pipeline import create_rag_chain

st.title("Pharma Medical Chatbot")

uploaded_file = st.file_uploader("Upload a medical file", type="pdf")

if uploaded_file:
    if not os.path.exists("data"):
        os.makedirs("data")
    
    file_path = os.path.join("data", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    qa_chain = create_rag_chain(file_path)
    
    query = st.text_input("Ask a question about the document")
    
    # Everything related to response must be inside this block
    if query:
        response = qa_chain.invoke({"query": query})
        
        st.subheader("Answer")
        st.write(response["result"])
        
        st.subheader("Sources")
        for doc in response["source_documents"]:
            st.write(doc.metadata)

        # Debug: show retrieved text
        st.subheader("Retrieved Context (Debug)")
        for i, doc in enumerate(response["source_documents"]):
            st.write(f"Chunk {i+1}")
            st.write(doc.page_content[:500])
            st.write("---")