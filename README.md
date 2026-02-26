# Medical PDF Chatbot (RAG-Based)

## Overview
The Medical PDF Chatbot is an AI-powered document question-answering system built using the Retrieval-Augmented Generation (RAG) approach. It allows users to upload medical or research PDF documents and ask questions in natural language.

The system retrieves relevant information from the document and generates answers strictly based on the available content. This reduces hallucinations and improves the accuracy and reliability of responses.

This project demonstrates a practical implementation of Large Language Models (LLMs) for intelligent document analysis and real-world AI applications.



## Key Features
- Upload medical or research PDF documents  
- Ask questions in natural language  
- Generates answers strictly based on document content  
- Displays source page information for verification  
- Provides a fallback response when information is not available  
- Interactive web interface using Streamlit  
- Secure API key management using environment variables  



## System Architecture

The application follows a standard Retrieval-Augmented Generation (RAG) pipeline:

### 1. Document Input
Users upload a PDF file through the Streamlit interface.

### 2. Document Processing
- Text extraction using PyPDFLoader  
- Content split into smaller chunks using RecursiveCharacterTextSplitter  

### 3. Embedding Generation
- HuggingFace embedding model: all-MiniLM-L6-v2  
- Converts text chunks into vector representations  

### 4. Vector Storage
- FAISS is used for efficient similarity search and storage of embeddings  

### 5. Retrieval
- Top relevant chunks are retrieved based on semantic similarity with the user query  

### 6. Answer Generation
- Retrieved context is passed to the Groq LLaMA model  
- The model generates answers using only the provided document context  



## Tech Stack
- Python  
- Streamlit  
- LangChain  
- Groq (LLaMA 3)  
- HuggingFace Embeddings  
- FAISS  
- PyPDF


## Design Highlights
- Retrieval-based architecture to reduce hallucinations  
- Efficient semantic search using FAISS  
- Prompt engineering to restrict answers to document context  
- Source page display for transparency  
- Modular and scalable code structure  



## Future Enhancements
- Support multiple PDF documents  
- Add chat history and conversational memory  
- Deploy on cloud platforms  
- Improve UI with chat-style interface  
- Use domain-specific medical embedding models  



## What This Project Demonstrates
- End-to-end implementation of Retrieval-Augmented Generation (RAG)  
- Integration of LangChain with vector databases  
- Practical use of LLMs for document intelligence  
- Prompt engineering and context control  
- Building real-world AI applications  