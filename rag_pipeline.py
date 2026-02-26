import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()



def create_rag_chain(pdf_path):
    ## Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    ## Split into chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=300)
    chunks=splitter.split_documents(documents)

    ## Create Embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"})

    vector_store=FAISS.from_documents(chunks,embeddings)
    retriever=vector_store.as_retriever(search_kwargs={"k":5})

    ## Groq LLm
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    llm = ChatGroq(
        groq_api_key=groq_key,
        model="llama-3.3-70b-versatile",
        temperature=0
)

    ##Prompt
    template = """
    You are a Medical AI Assistant.

    Answer only using the provided context.
    If the answer is not found in the context, say:
    "I cannot find this information in the document."

    Context:
    {context}

    Question:
    {question}
    """

    prompt=PromptTemplate(
        template=template,
        input_variables=["context","question"]
    )

    ## Rag Chain
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt":prompt},
        return_source_documents=True
        
    )
    return qa_chain

   
