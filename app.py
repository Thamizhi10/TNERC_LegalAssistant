import streamlit as st
import pickle
import numpy as np
import os
import zipfile
import requests
import fitz
import docx

from openai import OpenAI


# ---------------- OPENAI CLIENT ----------------
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set")
        st.stop()
    return OpenAI(api_key=api_key)


INDEX_ZIP_URL = "https://huggingface.co/tam3222/tnerc_index/resolve/main/index.zip"


# ---------------- DOWNLOAD ----------------
@st.cache_resource
def download_and_extract():
    if not os.path.exists("index"):
        r = requests.get(INDEX_ZIP_URL)

        if r.status_code != 200:
            raise Exception("Download failed")

        with open("index.zip", "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile("index.zip", 'r') as zip_ref:
            zip_ref.extractall()

    return True


# ---------------- LOAD (CRITICAL FIX) ----------------
@st.cache_resource
def load_chunks():
    with open("index/chunks.pkl", "rb") as f:
        rulings_chunks = pickle.load(f)

    with open("index/reg_chunks.pkl", "rb") as f:
        reg_chunks = pickle.load(f)

    return rulings_chunks, reg_chunks


# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    return ""


# ---------------- EMBEDDING ----------------
def get_embedding(text):
    if not text.strip():
        return np.zeros(1536)

    client = get_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


# ---------------- SEARCH ----------------
def simple_search(query_emb, chunks, top_k=3):
    scores = []

    for c in chunks:
        if "embedding" not in c or c["embedding"] is None:
            continue

        emb = np.array(c["embedding"])

        # Prevent crash due to mismatch
        if emb.shape != query_emb.shape:
            continue

        score = np.dot(query_emb, emb)
        scores.append((score, c))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scores[:top_k]]


def search_all(query, rulings_chunks, reg_chunks):
    q_emb = get_embedding(query)

    rulings = simple_search(q_emb, rulings_chunks)
    regs = simple_search(q_emb, reg_chunks)

    return rulings, regs


# ---------------- ANSWER GENERATION ----------------
def generate_answer(query, rulings, regs):

    context = "Regulations:\n"
    for r in regs:
        context += r["text"][:500] + "\n\n"

    context += "\nPast Rulings:\n"
    for r in rulings:
        context += r["text"][:500] + "\n\n"

    prompt = f"""
    You are an Electricity Ombudsman adjudicating a consumer dispute.

    Your role is to analyze the case, apply electricity regulations, and issue a reasoned order.

    Do NOT simply summarize. Deliver a structured adjudication.

    ---------------------
    CASE DETAILS:
    {query}
    ---------------------

    ---------------------
    RELEVANT REGULATIONS:
    {''.join([r["text"][:500] for r in regs])}
    ---------------------

    ---------------------
    REFERENCE RULINGS (for support only):
    {''.join([r["text"][:500] for r in rulings])}
    ---------------------

    INSTRUCTIONS:

    1. Identify the key facts of the case.
    2. Frame the issues that need determination.
    3. Examine applicable electricity regulations and explain their relevance.
    4. Apply the regulations to the facts of the case.
    5. Provide clear findings based on reasoning.
    6. Use past rulings only as supporting precedents, not as the primary basis.
    7. If regulations and rulings conflict, rely on regulations.
    8. Maintain formal, neutral, and authoritative tone.
    9. Avoid unnecessary verbosity.

    ---------------------

    OUTPUT FORMAT:

    Facts of the Case:
    (Summarize relevant facts concisely)

    Issues for Determination:
    (List key issues)

    Analysis:
    (Apply regulations to facts with reasoning)

    Findings:
    (Clearly state conclusions on each issue)

    Relevant Regulations:
    (Cite and explain applicable provisions)

    Supporting Precedents:
    (Only if relevant, with explanation)

    Final Order:
    (Clear directive — e.g., complaint allowed/rejected, relief granted, directions issued)
    """

    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ---------------- UI ----------------
st.title("TNERC Legal Assistant")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ---------- LOAD BUTTON ----------
if not st.session_state.data_loaded:
    if st.button("Load Knowledge Base"):
        with st.spinner("Downloading knowledge base..."):
            download_and_extract()

        with st.spinner("Loading data into memory..."):
            load_chunks()

        st.session_state.data_loaded = True
        st.success("Knowledge base loaded")


# ---------- MAIN APP ----------
if st.session_state.data_loaded:

    rulings_chunks, reg_chunks = load_chunks()

    uploaded_file = st.file_uploader("Upload case file", type=["pdf", "docx"])

    if uploaded_file is not None:

        text = extract_text(uploaded_file)
        query = text[:2000]

        if st.button("Analyze Case"):

            with st.spinner("Analyzing..."):
                rulings, regs = search_all(query, rulings_chunks, reg_chunks)
                answer = generate_answer(query, rulings, regs)

            st.subheader("Final Decision")
            st.write(answer)

            st.subheader("Relevant Regulations")
            for r in regs:
                st.write(r["text"][:500])

            st.subheader("Similar Past Rulings")
            for r in rulings:
                st.write(r["text"][:500])