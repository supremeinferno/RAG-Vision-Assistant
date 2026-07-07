# ==============================
# Python Standard Library
# ==============================
import os
import shutil
import tempfile
import base64

# ==============================
# Streamlit
# ==============================
import streamlit as st

# ==============================
# Environment Variables
# ==============================
from dotenv import load_dotenv
load_dotenv()

# ==============================
# LangChain Community
# ==============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# ==============================
# LangChain Text Splitter
# ==============================
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# Mistral AI
# ==============================
from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI,
)

# ==============================
# LangChain Prompt Template
# ==============================
from langchain_core.prompts import ChatPromptTemplate

# ==============================
# Local Project Files
# ==============================
from create_database import create_vector_database
from main import generate_response












# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Codex",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

BACKGROUND = "#0B111B"
CARD = "#111827"
GOLD = "#D4A44D"
TEXT = "#ECE8E1"
MUTED = "#9CA3AF"
BORDER = "#232C38"

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Cormorant+Garamond:wght@500;600;700&display=swap');

html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
    scroll-behavior:smooth;
}

body,.stApp{
    background:#0B111B;
    color:#ECE8E1;
}

#MainMenu,header,footer{
    visibility:hidden;
}

.block-container{
    max-width:1280px;
    padding-top:.8rem;
    padding-bottom:3rem;
}

::-webkit-scrollbar{width:8px;}
::-webkit-scrollbar-track{background:#101720;}
::-webkit-scrollbar-thumb{
    background:#2A3441;
    border-radius:20px;
}
::-webkit-scrollbar-thumb:hover{
    background:#D4A44D;
}

.navbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:20px 0;
    border-bottom:1px solid #232C38;
    margin-bottom:45px;
}

.logo{
    font-size:28px;
    font-weight:700;
    color:#ECE8E1;
}

.logo span{
    color:#D4A44D;
}

.nav-links{
    display:flex;
    gap:40px;
    align-items:center;
}

.nav-item{
    color:#98A2B3;
    font-size:15px;
    font-weight:500;
    cursor:pointer;
    transition:.25s;
}

.nav-item:hover{
    color:#D4A44D;
}

.hero{
    margin:60px 0;
}

.eyebrow{
    color:#D4A44D;
    font-size:12px;
    letter-spacing:4px;
    text-transform:uppercase;
    margin-bottom:18px;
}

.hero-title{
    font-family:"Cormorant Garamond",serif;
    font-size:72px;
    line-height:1.05;
    color:#ECE8E1;
    margin-bottom:18px;
}

.hero-title span{
    color:#D4A44D;
}

.hero-subtitle{
    font-size:18px;
    color:#98A2B3;
    max-width:760px;
    line-height:1.8;
}

.section-title{
    font-size:22px;
    font-weight:700;
    margin-bottom:18px;
}

# </style>
# """, unsafe_allow_html=True)



# ================= NAVBAR =================

st.markdown(
    """
    <h2 style="
    margin-bottom:10px;
    color:#ECE8E1;
    font-size:40px;
    font-weight:700px;
    letter-spacing:-0.5px;
    ">
    CODE<span style="color:#D4A44D;">X</span>
    </h2>
    """,
    unsafe_allow_html=True
)

st.divider()

# ================= HERO =================

st.markdown(
    """
<p style="
color:#D4A44D;
letter-spacing:4px;
font-size:12px;
font-weight:600;
margin-bottom:18px;
">
AN INSTRUMENT FOR CLOSE READING
</p>

<h1 style="
font-family:'Cormorant Garamond',serif;
font-size:84px;
line-height:1.05;
margin-bottom:18px;
color:#ECE8E1;
">
Converse with your
<span style="color:#D4A44D;">documents</span><br>
interrogate your
<span style="color:#D4A44D;">images</span>.
</h1>

<p style="
color:#9CA3AF;
font-size:18px;
max-width:820px;
line-height:1.8;
">
Codex assembles a private library from your PDFs and images and answers with grounded citations using a multimodal RAG pipeline.
</p>
""",
    unsafe_allow_html=True,
)

# -------- END OF PART 1 --------







# =====================================================
# PART 2 - MAIN LAYOUT + LIBRARY PANEL
# Paste below Part 1
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([0.9, 2.1], gap="large")

# =====================================================
# LIBRARY PANEL
# =====================================================

with left:

    st.markdown(
        '<div class="section-title">Library</div>',
        unsafe_allow_html=True
    )

    st.markdown("##### Documents")

    pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_upload"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("##### Images")

    image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="image_upload"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if image:
        st.image(image, use_container_width=True)
    else:
        st.info("Image preview will appear here.")

    st.markdown("<br>", unsafe_allow_html=True)

    create_db = st.button(
        "Build Library",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.database_ready:
        st.success("Knowledge Base Ready")
    else:
        st.caption("No knowledge base created.")





# =====================================================
# PART 3 - DIALOGUE PANEL (UI ONLY)
# Paste below Part 2
# =====================================================

with right:

    st.markdown(
        '<div class="section-title">Dialogue</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
        <div style="
        background:#111827;
        border:1px solid #232C38;
        border-radius:18px;
        padding:28px;
        margin-bottom:20px;">

        <h2 style="margin:0;color:#ECE8E1;">
        Welcome to Codex
        </h2>

        <p style="margin-top:12px;color:#9CA3AF;line-height:1.8;">
        Upload your PDF from the Library panel and start asking questions.
        </p>

        </div>
        """, unsafe_allow_html=True)

    if len(st.session_state.messages) == 0:

        st.markdown("##### Suggested Questions")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Summarize Document", use_container_width=True):
                st.session_state["preset"] = "Summarize this document."

            if st.button("Extract Key Points", use_container_width=True):
                st.session_state["preset"] = "Extract the key points."

        with c2:
            if st.button("Generate Notes", use_container_width=True):
                st.session_state["preset"] = "Generate concise notes."

            if st.button("Explain Diagrams", use_container_width=True):
                st.session_state["preset"] = "Explain the diagrams."

        st.divider()

    chat_area = st.container(height=560)

    with chat_area:

        if len(st.session_state.messages) == 0:

            st.info("No conversation yet.")

        else:

            for msg in st.session_state.messages:

                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    preset = st.session_state.get("preset", "")

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question is None and preset:
        question = preset
        st.session_state["preset"] = ""








# =====================================================
# PART 4 - BACKEND INTEGRATION
# Paste below Part 3
# =====================================================

# -----------------------------
# Build Knowledge Base
# -----------------------------

if create_db:

    if pdf is None:

        st.warning("Please upload a PDF first.")

    else:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(pdf.getvalue())
            pdf_path = tmp.name

        with st.spinner("Building knowledge base..."):

            try:

                pages, chunks = create_vector_database(pdf_path)

                st.session_state.database_ready = True

                st.success("Knowledge Base Created!")

                m1, m2 = st.columns(2)

                with m1:
                    st.metric("Pages", pages)

                with m2:
                    st.metric("Chunks", chunks)

            except Exception as e:

                st.error(str(e))

            finally:

                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

# -----------------------------
# Ask Question
# -----------------------------

if question:

    if not st.session_state.database_ready:

        st.warning(
            "Please build the knowledge base first."
        )

    else:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.spinner("Searching your documents..."):

            try:

                answer, docs = generate_response(
                    question=question,
                    image=image,
                    response_style="⚖️ Balanced",
                    answer_length="Medium"
                )

            except Exception as e:

                answer = f"Error: {e}"
                docs = []

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        st.session_state["retrieved_docs"] = docs

        st.rerun()

# -----------------------------
# Sources
# -----------------------------

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if st.session_state.retrieved_docs:

    with st.expander("Retrieved Sources", expanded=False):

        for i, doc in enumerate(
            st.session_state.retrieved_docs,
            start=1
        ):

            page = doc.metadata.get("page", "?")

            st.markdown(
                f"### Source {i} · Page {page}"
            )

            st.write(doc.page_content)

            st.divider()





# =====================================================
# PART 5 - FINAL UI POLISH
# Paste below Part 4
# =====================================================

st.markdown("""
<style>

/* ---------- Cards ---------- */

[data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid #2A3441 !important;
    border-radius:18px !important;
    background:#111827 !important;
    transition:.25s;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover{
    border-color:#D4A44D !important;
    transform:translateY(-2px);
}

/* ---------- Buttons ---------- */

.stButton>button{
    width:100%;
    height:46px;
    border-radius:12px;
    border:1px solid #D4A44D;
    background:#0F1621;
    color:#ECE8E1;
    font-weight:600;
}

.stButton>button:hover{
    background:#D4A44D;
    color:#111827;
}

/* ---------- Upload ---------- */

[data-testid="stFileUploaderDropzone"]{
    border-radius:16px;
    border:1.5px dashed #D4A44D55;
    background:#101720;
    min-height:170px;
}

/* ---------- Chat ---------- */

[data-testid="stChatMessage"]{
    border-radius:16px;
    border:1px solid #2A3441;
    padding:10px;
    margin-bottom:10px;
    background:#111827;
}

/* ---------- Metrics ---------- */

[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #2A3441;
    border-radius:14px;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:25px 0;border-top:1px solid #232C38;margin-top:40px;">

<h3 style="margin:0;color:#ECE8E1;font-family:'Cormorant Garamond',serif;">
Codex
</h3>

<p style="color:#9CA3AF;margin-top:6px;">
Multimodal Retrieval Assistant
</p>

</div>
""", unsafe_allow_html=True)










# =====================================================
# PART 6 - FINAL CLEANUP & UX IMPROVEMENTS
# Paste below Part 5
# =====================================================

# ---------- Page spacing ----------

st.markdown("""
<style>

.block-container{
    max-width:1600px !important;
    padding-left:3rem !important;
    padding-right:3rem !important;
}

/* Hero */

.hero-title{
    font-size:84px !important;
    max-width:1100px !important;
}

.hero-subtitle{
    font-size:19px !important;
    max-width:900px !important;
}

/* Section Titles */

.section-title{
    font-size:24px !important;
    font-weight:700 !important;
    letter-spacing:.2px;
}

/* Chat */

[data-testid="stChatMessage"]{
    margin-bottom:16px !important;
    border-radius:18px !important;
}

[data-testid="stChatInput"]{
    position:sticky;
    bottom:0;
}

/* Image */

img{
    border-radius:18px !important;
}

/* Buttons */

.stButton>button{
    transition:all .2s ease-in-out !important;
}

.stButton>button:hover{
    transform:translateY(-2px);
}

/* Success / Warning */

.stSuccess,
.stInfo,
.stWarning,
.stError{
    border-radius:14px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- Top Status ----------

status1, status2, status3 = st.columns(3)

with status1:
    st.caption("System")
    if st.session_state.database_ready:
        st.success("Ready")
    else:
        st.info("Waiting")

with status2:
    st.caption("Document")
    if pdf:
        st.success(pdf.name)
    else:
        st.info("Not Uploaded")

with status3:
    st.caption("Image")
    if image:
        st.success("Uploaded")
    else:
        st.info("Optional")




