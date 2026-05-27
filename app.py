import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.title("RAG Document Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa" not in st.session_state:
    st.session_state.qa = None

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.qa is None:
    all_chunks = []
    with st.spinner("Processing PDFs..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(all_chunks, embeddings)
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])

        prompt_template = """Use the following context to answer the question.
Always cite which part of the document your answer comes from.
If you don't know the answer, say so.

Context:
{context}

Question: {question}
Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 20}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    st.success(f"Ready! Loaded {len(uploaded_files)} PDF(s).")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.qa:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa.invoke({"query": question})
                answer = result["result"]
                sources = result["source_documents"]

                st.write(answer)

                with st.expander("Sources"):
                    for i, doc in enumerate(sources[:3]):
                        st.caption(f"Chunk {i+1} (page {doc.metadata.get('page', '?')+1}): {doc.page_content[:200]}...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})