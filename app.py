import os
import tempfile
import base64
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

st.set_page_config(page_title="RAG Document Assistant", layout="centered")

def load_asset(path):
    with open(path, "rb") as f:
        data = f.read()
    ext = path.split(".")[-1]
    mime = {"gif": "image/gif", "avif": "image/avif", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

flower_bg = load_asset("grass_field.jpg")
doggy = load_asset("doggy_transparent2.gif")
ball = load_asset("ball.png")

st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: #0f1117 !important;
        color: #e8f7ff !important;
    }}
    [data-testid="stMain"] {{
        background-color: #0f1117 !important;
    }}
    [data-testid="stHeader"] {{
        background-color: #0f1117 !important;
    }}
    p, span, label, div, h1, h2, h3 {{
        color: #e8f7ff !important;
    }}
    .stChatMessage {{
        background-color: #1e2130 !important;
        border-radius: 16px !important;
        border: 1px solid #2d4a6e !important;
    }}
    .stButton > button {{
        border-radius: 20px !important;
        background-color: #1e2130 !important;
        border: 1.5px solid #378add !important;
        color: #85b7eb !important;
        font-size: 13px !important;
    }}
    .stButton > button:hover {{
        background-color: #2d4a6e !important;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid #2d4a6e !important;
        border-radius: 12px !important;
        background-color: #1e2130 !important;
    }}
    [data-testid="stFileUploader"] {{
        background-color: #1e2130 !important;
        border-radius: 12px !important;
        border: 1.5px dashed #378add !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: #1e2130 !important;
        color: #e8f7ff !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: #2d4a6e !important;
        color: #85b7eb !important;
        border: 1px solid #378add !important;
    }}
    .stSuccess {{
        background-color: #1a3a2a !important;
        color: #6fcf97 !important;
        border-radius: 10px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: #1e2130 !important;
        color: #e8f7ff !important;
        border-radius: 20px !important;
        border: 1.5px solid #378add !important;
    }}
    .banner {{
        position: relative;
        width: 100%;
        height: 210px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 20px;
    }}
    .banner img.bg {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
    .banner img.dog {{
        position: absolute;
        height: 200px;
        bottom: 6px;
        animation: walkacross 10s linear infinite;
    }}
    @keyframes walkacross {{
        from {{ left: -80px; }}
        to {{ left: 105%; }}
    }}
    .app-title {{
        font-size: 26px;
        font-weight: 600;
        color: #85b7eb !important;
        margin-bottom: 20px;
    }}
    .doggy-ball-loader {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0;
    }}
    .doggy-ball-loader img.ball {{
        width: 28px;
        animation: bounce 0.4s ease-in-out infinite alternate;
    }}
    .doggy-ball-loader span {{
        color: #85b7eb !important;
    }}
    @keyframes bounce {{
        from {{ transform: translateY(0px); }}
        to {{ transform: translateY(-10px); }}
    }}
</style>

<div class="banner">
    <img class="bg" src="{flower_bg}" />
    <img class="dog" src="{doggy}" />
</div>

<p class="app-title">RAG Document Assistant</p>
""", unsafe_allow_html=True)

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
    with st.spinner(""):
        st.markdown(f'<div class="doggy-ball-loader"><img class="ball" src="{ball}" /><span>processing your PDF...</span></div>', unsafe_allow_html=True)
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

    st.success(f"ready! loaded {len(uploaded_files)} PDF(s).")

if st.session_state.summary:
    with st.expander("document summary", expanded=True):
        st.write(st.session_state.summary)

if st.session_state.suggestions:
    st.write("**suggested questions:**")
    cols = st.columns(3)
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.pending_question = suggestion

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.qa:
    question = st.chat_input("ask something about your document...")

    if "pending_question" in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(""):
                st.markdown(f'<div class="doggy-ball-loader"><img class="ball" src="{ball}" /><span>thinking...</span></div>', unsafe_allow_html=True)
                result = st.session_state.qa.invoke({"question": question})
                answer = result["answer"]
                sources = result["source_documents"]

            st.write(answer)

            with st.expander("sources"):
                for i, doc in enumerate(sources[:3]):
                    st.caption(f"chunk {i+1} (page {doc.metadata.get('page', '?')+1}): {doc.page_content[:200]}...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})