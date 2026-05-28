import os
import tempfile
import base64
import streamlit as st
import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

st.set_page_config(page_title="PupQuest", layout="wide")

def load_asset(path):
    with open(path, "rb") as f:
        data = f.read()
    ext = path.split(".")[-1]
    mime = {"gif": "image/gif", "avif": "image/avif", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

flower_bg = load_asset("grassy_field.jpg")
doggy = load_asset("doggy_transparent2.gif")
ball = load_asset("ball.png")

USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🐾"

st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: #0f1117 !important;
        color: #e8f7ff !important;
    }}
    [data-testid="stMain"] {{ background-color: #0f1117 !important; }}
    [data-testid="stHeader"] {{ background-color: #0f1117 !important; }}
    p, span, label, div, h1, h2, h3 {{ color: #e8f7ff !important; }}
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
    .stButton > button:hover {{ background-color: #2d4a6e !important; }}
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
        height: 200px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 20px;
    }}
    .banner img.bg {{ width: 100%; height: 100%; object-fit: cover; object-position: center; }}
    .banner img.dog {{
        position: absolute;
        height: 150px;
        bottom: 4px;
        animation: walkacross 10s linear infinite;
    }}
    @keyframes walkacross {{
        from {{ left: -80px; }}
        to {{ left: 105%; }}
    }}
    .app-title {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #85b7eb, #6fcf97, #85b7eb);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
    }}
    @keyframes shimmer {{
        from {{ background-position: 0% center; }}
        to {{ background-position: 200% center; }}
    }}
    .doc-pill {{
        display: inline-block;
        background-color: #2d4a6e;
        color: #85b7eb !important;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        margin: 2px;
    }}
    .chunk-stat {{
        font-size: 12px;
        color: #6fcf97 !important;
        margin-top: 4px;
    }}
    .source-pill {{
        display: inline-block;
        background-color: #1a2d45;
        border: 1px solid #378add;
        color: #85b7eb !important;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 11px;
        margin: 2px;
        cursor: default;
    }}
    .followup-label {{
        font-size: 12px;
        color: #6b7fa3 !important;
        margin: 8px 0 4px;
    }}
    .game-card {{
        background-color: #1e2130;
        border: 1.5px solid #378add;
        border-radius: 16px;
        padding: 16px;
        margin: 8px 0;
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
    .doggy-ball-loader span {{ color: #85b7eb !important; }}
    @keyframes bounce {{
        from {{ transform: translateY(0px); }}
        to {{ transform: translateY(-10px); }}
    }}
</style>

<div class="banner">
    <img class="bg" src="{flower_bg}" />
    <img class="dog" src="{doggy}" />
</div>
<p class="app-title">PupQuest 🐾</p>
""", unsafe_allow_html=True)

def reset():
    for key in ["chat_history", "qa", "summary", "suggestions", "doc_names",
                "chunk_count", "pdf_bytes", "pdf_page_count", "doc_type", "game_mode"]:
        if key in st.session_state:
            del st.session_state[key]

def classify_document(text, llm):
    prompt = f"""Classify this document into exactly one of these categories:
- lecture: academic lectures, textbooks, study material, science, history
- medical: nursing, psychology, therapy, diagnosis, health, biology
- finance: trading, investing, business, economics, money
- howto: manuals, guides, instructions, recipes, tutorials
- other: anything that doesn't fit above

Return only the single word category, nothing else.

Document:
{text[:3000]}"""
    result = llm.invoke(prompt).content.strip().lower()
    for cat in ["lecture", "medical", "finance", "howto"]:
        if cat in result:
            return cat
    return "other"

for key, default in [("chat_history", []), ("qa", None), ("summary", None),
                     ("suggestions", []), ("doc_names", []), ("chunk_count", 0),
                     ("pdf_bytes", None), ("pdf_page_count", 1),
                     ("doc_type", None), ("game_mode", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# top bar
if st.session_state.doc_names:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        pills = " ".join([f'<span class="doc-pill">📄 {n}</span>' for n in st.session_state.doc_names])
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown(f'<p class="chunk-stat">✦ {st.session_state.chunk_count} chunks indexed</p>', unsafe_allow_html=True)
    with col2:
        if st.session_state.chat_history:
            chat_text = "\n\n".join([f"{'You' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state.chat_history])
            st.download_button("⬇ export", chat_text, file_name="chat_export.txt", mime="text/plain")
    with col3:
        if st.button("🔄 reset"):
            reset()
            st.rerun()

# upload
if not st.session_state.qa:
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        all_chunks = []
        with st.spinner(""):
            st.markdown(f'<div class="doggy-ball-loader"><img class="ball" src="{ball}" /><span>sniffing your document...</span></div>', unsafe_allow_html=True)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(file_bytes)
                    tmp_path = f.name
                if i == 0:
                    st.session_state.pdf_bytes = file_bytes
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    st.session_state.pdf_page_count = len(doc)
                    doc.close()
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)

            st.session_state.doc_names = [f.name for f in uploaded_files]
            st.session_state.chunk_count = len(all_chunks)

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
            st.session_state.summary = llm.invoke(f"Summarize this document in 3 sentences:\n\n{full_text}").content
            st.session_state.doc_type = classify_document(full_text, llm)

            raw = llm.invoke(f"Given this document, suggest exactly 3 short questions a user might ask. Return only a numbered list, nothing else:\n\n{full_text[:2000]}").content
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            st.session_state.suggestions = [l.lstrip("123. ").strip() for l in lines[:3]]

        st.success(f"ready! loaded {len(uploaded_files)} PDF(s), {st.session_state.chunk_count} chunks indexed.")
        st.rerun()

# main layout
if st.session_state.qa:
    left_col, right_col = st.columns([1, 1], gap="medium")

    # LEFT — PDF preview
    with left_col:
        st.markdown("**document preview**")
        if st.session_state.pdf_bytes:
            page_num = st.number_input(
                "page",
                min_value=1,
                max_value=st.session_state.pdf_page_count,
                value=1,
                step=1
            ) - 1
            doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
            page = doc[page_num]
            pix = page.get_pixmap(dpi=130)
            img_bytes = pix.tobytes("png")
            doc.close()
            st.image(img_bytes, use_container_width=True)

        if st.session_state.summary:
            with st.expander("document summary", expanded=False):
                st.write(st.session_state.summary)

    # RIGHT — mode selector + chat
    with right_col:

        # game mode selector
        if st.session_state.game_mode is None:
            type_labels = {
                "lecture": "📚 lecture / study material",
                "medical": "🏥 medical / psychology",
                "finance": "📈 finance / business",
                "howto": "🔧 how-to / manual",
                "other": "📄 general document"
            }
            game_options = {
                "lecture": ("🧠 quiz mode", "🩺 role play mode", "🎲 decision game mode", "🎯 step challenge mode", "📖 chat mode"),
                "medical": ("🩺 role play mode", "🧠 quiz mode", "🎲 decision game mode", "🎯 step challenge mode", "📖 chat mode"),
                "finance": ("🎲 decision game mode", "🧠 quiz mode", "🩺 role play mode", "🎯 step challenge mode", "📖 chat mode"),
                "howto":   ("🎯 step challenge mode", "🧠 quiz mode", "🩺 role play mode", "🎲 decision game mode", "📖 chat mode"),
                "other":   ("📖 chat mode", "🧠 quiz mode", "🩺 role play mode", "🎲 decision game mode", "🎯 step challenge mode"),
            }
            doc_type = st.session_state.doc_type or "other"
            detected = type_labels.get(doc_type, "📄 general document")
            st.markdown(f'<div class="game-card"><p style="margin:0;font-size:13px;color:#6fcf97 !important;">🐾 pup detected: <strong>{detected}</strong></p><p style="margin:4px 0 0;font-size:12px;color:#6b7fa3 !important;">choose your mode:</p></div>', unsafe_allow_html=True)

            options = game_options.get(doc_type, ("📖 chat mode", "🧠 quiz mode"))
            mode_cols = st.columns(min(len(options), 3))
            for i, opt in enumerate(options):
                if mode_cols[i % 3].button(opt, key=f"mode_{i}"):
                    st.session_state.game_mode = opt
                    st.rerun()

        else:
            mode = st.session_state.game_mode
            st.markdown(f'**{mode}**')

            if st.button("↩ switch mode", key="switch_mode"):
                st.session_state.game_mode = None
                st.session_state.chat_history = []
                st.rerun()

            # inject system prompt based on mode
            if not st.session_state.chat_history:
                doc_type = st.session_state.doc_type or "other"
                llm_init = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
                full_text = ""
                if st.session_state.pdf_bytes:
                    doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
                    for pg in doc:
                        full_text += pg.get_text()
                        if len(full_text) > 3000:
                            break
                    doc.close()

                if "quiz" in mode:
                    opener = llm_init.invoke(f"You are a quiz master. Based on this document, ask the user one multiple choice question to start. Format it clearly with A/B/C/D options. Document:\n\n{full_text[:3000]}").content
                elif "role play" in mode:
                    opener = llm_init.invoke(f"You are a patient with symptoms described in this document. Start the roleplay by introducing yourself and describing your main complaint naturally, as a real patient would. The user is the clinician. Document:\n\n{full_text[:3000]}").content
                elif "decision game" in mode:
                    opener = llm_init.invoke(f"You are a financial game master. Create a short scenario based on this document where the user must make a financial decision. Give them 2-3 options. Document:\n\n{full_text[:3000]}").content
                elif "step challenge" in mode:
                    opener = llm_init.invoke(f"You are a guide. Based on this document, present the user with the first step of a process and ask them what comes next. Document:\n\n{full_text[:3000]}").content
                else:
                    opener = None

                if opener:
                    st.session_state.chat_history.append({"role": "assistant", "content": opener})

            # chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar=BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        pills = " ".join([f'<span class="source-pill">📎 p.{s}</span>' for s in message["sources"][:5]])
                        st.markdown(f'{pills} <span style="font-size:11px;color:#6b7fa3 !important;">— {message.get("matched",0)} chunks matched</span>', unsafe_allow_html=True)
                    if message["role"] == "assistant" and "followups" in message:
                        st.markdown('<p class="followup-label">✨ follow-up suggestions:</p>', unsafe_allow_html=True)
                        fu_cols = st.columns(len(message["followups"]))
                        for j, fu in enumerate(message["followups"]):
                            if fu_cols[j].button(fu, key=f"fu_{message.get('id',0)}_{j}"):
                                st.session_state.pending_question = fu

            question = st.chat_input("your response...")

            if "pending_question" in st.session_state and st.session_state.pending_question:
                question = st.session_state.pending_question
                st.session_state.pending_question = None

            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.write(question)

                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    with st.spinner(""):
                        st.markdown(f'<div class="doggy-ball-loader"><img class="ball" src="{ball}" /><span>fetching...</span></div>', unsafe_allow_html=True)
                        if "chat" in mode:
                            result = st.session_state.qa.invoke({"question": question})
                            answer = result["answer"]
                            sources = result["source_documents"]
                        else:
                            history_text = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state.chat_history[:-1]])
                            game_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
                            if "quiz" in mode:
                                system = "You are a quiz master. Grade the user's answer, explain why it's correct or incorrect, then ask a new question with A/B/C/D options. Stay in quiz mode always."
                            elif "role play" in mode:
                                system = "You are a patient. Stay in character always. Respond to the clinician's questions as a real patient would. Never break character."
                            elif "decision" in mode:
                                system = "You are a financial game master. Evaluate the user's decision, explain the consequences, then present a new scenario with options. Stay in game mode always."
                            else:
                                system = "You are a step-by-step guide. Evaluate the user's answer, then present the next step as a challenge. Stay in game mode always."
                            answer = game_llm.invoke(f"{system}\n\nConversation so far:\n{history_text}\n\nUser: {question}").content
                            sources = []
                    st.write(answer)

                    if "chat" in mode:
                        pages = sorted(set([doc.metadata.get("page", 0) + 1 for doc in sources]))[:5]
                        matched = len(set([d.page_content[:50] for d in sources]))
                        pills_html = " ".join([f'<span class="source-pill">📎 p.{p}</span>' for p in pages])
                        st.markdown(f'{pills_html} <span style="font-size:11px;color:#6b7fa3 !important;">— {matched} chunks matched</span>', unsafe_allow_html=True)

                        llm_fu = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
                        fu_raw = llm_fu.invoke(f"Given this answer, suggest 2 short follow-up questions. Return only a numbered list:\n\n{answer}").content
                        fu_lines = [l.strip() for l in fu_raw.strip().split("\n") if l.strip()]
                        followups = [l.lstrip("12. ").strip() for l in fu_lines[:2]]
                        st.markdown('<p class="followup-label">✨ follow-up suggestions:</p>', unsafe_allow_html=True)
                        fu_cols = st.columns(2)
                        msg_id = len(st.session_state.chat_history)
                        for j, fu in enumerate(followups):
                            if fu_cols[j].button(fu, key=f"fu_new_{j}"):
                                st.session_state.pending_question = fu

                        st.session_state.chat_history.append({
                            "role": "assistant", "content": answer,
                            "sources": pages, "matched": matched,
                            "followups": followups, "id": msg_id
                        })
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})