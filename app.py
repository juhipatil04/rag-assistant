import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

st.title("RAG Document Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa" not in st.session_state:
    st.session_state.qa = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

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

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 20}),
            memory=memory,
            return_source_documents=True
        )

        full_text = " ".join([c.page_content for c in all_chunks[:20]])
        summary_prompt = f"Summarize this document in 3 sentences:\n\n{full_text}"
        st.session_state.summary = llm.invoke(summary_prompt).content

        suggestions_prompt = f"Given this document, suggest exactly 3 short questions a user might ask. Return only a numbered list, nothing else:\n\n{full_text[:2000]}"
        raw = llm.invoke(suggestions_prompt).content
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        st.session_state.suggestions = [l.lstrip("123. ").strip() for l in lines[:3]]

if st.session_state.summary:
    with st.expander("Document Summary", expanded=True):
        st.write(st.session_state.summary)

if st.session_state.suggestions:
    st.write("**Suggested questions:**")
    cols = st.columns(3)
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.pending_question = suggestion

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.qa:
    question = st.chat_input("Ask a question about your documents...")

    if "pending_question" in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa.invoke({"question": question})
                answer = result["answer"]
                sources = result["source_documents"]

                st.write(answer)

                with st.expander("Sources"):
                    for i, doc in enumerate(sources[:3]):
                        st.caption(f"Chunk {i+1} (page {doc.metadata.get('page', '?')+1}): {doc.page_content[:200]}...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})