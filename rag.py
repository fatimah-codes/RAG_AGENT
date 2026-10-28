import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------
# Page setup
# ---------------------------------------------------
st.set_page_config(page_title="PDF RAG Agent", page_icon="📄", layout="wide")

# ---------------------------------------------------
# Theme + animations
# ---------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #1b0a3c, #3a1c71, #0f4c81, #0b7a75, #6a1b9a);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
    color: #f4f1ff;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating glowing orbs */
.orb { position: fixed; border-radius: 50%; filter: blur(60px); opacity: .45;
       z-index: 0; pointer-events: none; animation: floaty 14s ease-in-out infinite; }
.orb.o1 { width: 320px; height: 320px; background: #ff6ec7; top: -80px; left: -60px; }
.orb.o2 { width: 380px; height: 380px; background: #4facfe; bottom: -120px; right: -80px; animation-delay: -4s; }
.orb.o3 { width: 260px; height: 260px; background: #43e97b; top: 45%; left: 55%; animation-delay: -8s; opacity: .3; }
@keyframes floaty {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%      { transform: translate(40px, -50px) scale(1.12); }
    66%      { transform: translate(-30px, 40px) scale(.94); }
}

/* Hide default chrome */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Main container above the orbs */
.block-container { position: relative; z-index: 1; padding-top: 2rem; animation: fadeUp .8s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Hero title */
.hero { text-align: center; margin: 1rem 0 2rem; animation: fadeUp 1s ease both; }
.hero h1 {
    font-size: 3rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #ff9a9e, #fad0c4, #a1c4fd, #c2ffd8, #ff9a9e);
    background-size: 300% 100%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
}
@keyframes shine { to { background-position: 300% 0; } }
.hero p { opacity: .8; font-weight: 300; letter-spacing: .5px; }
.bounce { display: inline-block; animation: bounce 2.4s ease-in-out infinite; }
@keyframes bounce { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }

/* Glass cards (forms, expanders, uploader) */
[data-testid="stForm"], [data-testid="stFileUploader"], [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
    padding: .6rem;
    animation: fadeUp .9s ease both;
}
[data-testid="stForm"]:hover, [data-testid="stFileUploader"]:hover {
    box-shadow: 0 14px 50px rgba(124, 77, 255, .45);
    transition: box-shadow .4s ease;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 8, 40, 0.65);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}

/* Inputs */
.stTextInput input {
    background: rgba(255,255,255,.1) !important; color: #fff !important;
    border: 1px solid rgba(255,255,255,.25) !important; border-radius: 12px !important;
    transition: all .3s ease;
}
.stTextInput input:focus { border-color: #a18cd1 !important; box-shadow: 0 0 0 3px rgba(161,140,209,.35) !important; }

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #ff6ec7, #7c4dff, #4facfe);
    background-size: 200% 200%;
    color: #fff; border: none; border-radius: 14px; font-weight: 600;
    padding: .6rem 1.4rem; width: 100%;
    transition: transform .25s ease, box-shadow .25s ease, background-position .6s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 28px rgba(124, 77, 255, .55);
    background-position: 100% 0;
    color: #fff;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 18px; padding: 1rem; margin-bottom: .8rem;
    backdrop-filter: blur(10px);
    animation: slideIn .5s cubic-bezier(.2,.8,.2,1) both;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-30px) scale(.97); }
    to   { opacity: 1; transform: translateX(0) scale(1); }
}
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stChatInput"] textarea { color: #fff; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 14px; animation: fadeUp .6s ease both; }

/* Pulse badge */
.badge { display:inline-block; padding:.3rem .9rem; border-radius:999px; font-size:.8rem;
         background: rgba(67,233,123,.2); border:1px solid rgba(67,233,123,.6);
         animation: pulse 2s ease-in-out infinite; }
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(67,233,123,.5); }
    50%     { box-shadow: 0 0 0 10px rgba(67,233,123,0); }
}

/* ---------- Readability fixes ---------- */
.emoji { font-size: 3rem; text-align: center; line-height: 1.2; }

/* All labels and text on dark background */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] p,
.stApp label { color: #f4f1ff !important; }

/* Slider numbers */
[data-testid="stThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
    color: #ffffff !important; font-weight: 600;
}

/* Selectbox as dark glass */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,.14) !important;
    border: 1px solid rgba(255,255,255,.3) !important;
    border-radius: 12px !important;
}
[data-baseweb="select"] * { color: #ffffff !important; }
[data-baseweb="select"] svg { fill: #ffffff !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.12) !important;
    border: 2px dashed rgba(255,255,255,.45) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small { color: #ffffff !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #ff6ec7, #7c4dff) !important;
    color: #ffffff !important; border: none !important; border-radius: 10px !important;
}

/* Alerts (info / success / warning / error) */
[data-testid="stAlert"] {
    background: rgba(255,255,255,.16) !important;
    border: 1px solid rgba(255,255,255,.35) !important;
}
[data-testid="stAlert"] *, [data-testid="stAlert"] p { color: #ffffff !important; }

/* Badge */
.badge { color: #eafff1 !important; font-weight: 500; }

/* Expander text + sources */
[data-testid="stExpander"] *, [data-testid="stChatMessage"] * { color: #f4f1ff; }
</style>

<div class="orb o1"></div>
<div class="orb o2"></div>
<div class="orb o3"></div>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------
# Step 1: API key gate (shown first)
# ---------------------------------------------------
def validate_key(key: str):
    """Return (ok, message). Makes one tiny call to confirm the key works."""
    from openai import OpenAI

    try:
        OpenAI(api_key=key).models.list()
        return True, ""
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg or "Incorrect API key" in msg:
            return False, "This API key is invalid. Please check it and try again."
        return False, f"Could not verify the key: {msg[:200]}"


def api_key_screen():
    # hide sidebar on this screen
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="hero">
            <div class="emoji bounce">🔐</div>
            <h1>Welcome</h1>
            <p>Enter your OpenAI API key to get started</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("api_key_form"):
            key = st.text_input(
                "OpenAI API Key", type="password", placeholder="sk-..."
            )
            go = st.form_submit_button("🚀 Continue")
        st.caption("Your key stays in this session only and is never saved.")

        if go:
            if not key.strip():
                st.warning("Please enter your API key first.")
            else:
                with st.spinner("Verifying your key…"):
                    ok, msg = validate_key(key.strip())
                if ok:
                    st.session_state.api_key = key.strip()
                    st.session_state.key_ok = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error(msg)


if not st.session_state.get("key_ok"):
    api_key_screen()
    st.stop()

# ---------------------------------------------------
# Step 2: Main console
# ---------------------------------------------------
os.environ["OPENAI_API_KEY"] = st.session_state.api_key

st.markdown(
    """
    <div class="hero">
        <div class="emoji bounce">📄</div>
        <h1>PDF RAG Agent</h1>
        <p>Upload a PDF, then ask questions - answers come only from the document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown('<span class="badge">● API key connected</span>', unsafe_allow_html=True)
    st.write("")
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)
    st.divider()
    if st.button("🔑 Change API key"):
        for k in ("key_ok", "api_key", "vectorstore", "file_sig", "messages"):
            st.session_state.pop(k, None)
        st.rerun()

uploaded_pdf = st.file_uploader("📎 Upload your PDF", type=["pdf"])


def build_vectorstore(pdf_bytes: bytes, size: int, overlap: int) -> FAISS:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


if uploaded_pdf is not None:
    file_sig = (uploaded_pdf.name, uploaded_pdf.size, chunk_size, chunk_overlap)
    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("Reading and indexing PDF…"):
            st.session_state.vectorstore = build_vectorstore(
                uploaded_pdf.getvalue(), chunk_size, chunk_overlap
            )
        st.session_state.file_sig = file_sig
        st.session_state.messages = []
    st.success(f"Indexed **{uploaded_pdf.name}** – ask away!")

# ---------------------------------------------------
# RAG chain
# ---------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions strictly using the "
     "provided context from a PDF document.\n"
     "Rules:\n"
     "1. Answer ONLY from the context below.\n"
     "2. If the answer is not in the context, say: "
     "\"I couldn't find that in the document.\"\n"
     "3. Cite the page number(s) you used, e.g. (p. 3).\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {d.metadata['page'] + 1 if isinstance(d.metadata.get('page'), int) else '?'}]\n"
        f"{d.page_content}"
        for d in docs
    )


# ---------------------------------------------------
# Chat interface
# ---------------------------------------------------
if "vectorstore" in st.session_state:
    st.session_state.setdefault("messages", [])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the PDF...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        llm = ChatOpenAI(model=model_name, temperature=0)
        chain = RAG_PROMPT | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = retriever.invoke(question)  # retrieve once, reuse for sources
                answer = chain.invoke({"context": format_docs(docs), "question": question})
                st.markdown(answer)

            with st.expander("🔍 Sources (retrieved chunks)"):
                for doc in docs:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1 if isinstance(page, int) else page}**")
                    st.text(doc.page_content[:500])
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a PDF to get started.")