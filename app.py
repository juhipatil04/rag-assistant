import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

st.title("RAG Document Assistant")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(chunks, embeddings)
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 20}))

    st.success("Ready! Ask a question below.")

    question = st.text_input("Your question:")
    if question:
        with st.spinner("Thinking..."):
            result = qa.invoke({"query": question})
        st.write(result["result"])