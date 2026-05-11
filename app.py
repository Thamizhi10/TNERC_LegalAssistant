import streamlit as st
import pickle
import numpy as np
import os
import zipfile
import requests
import fitz
import docx

from openai import OpenAI

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- OPENAI CLIENT ----------------
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set")
        st.stop()
    return OpenAI(api_key=api_key)


INDEX_ZIP_URL = "https://huggingface.co/tam3222/tnerc_index/resolve/main/indexv2.zip"


# ---------------- DOWNLOAD ----------------
@st.cache_resource
def download_and_extract():
    if not os.path.exists("index"):
        r = requests.get(INDEX_ZIP_URL)

        if r.status_code != 200:
            raise Exception("Download failed")

        with open("indexv2.zip", "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile("indexv2.zip", 'r') as zip_ref:
            zip_ref.extractall()

    return True


# ---------------- LOAD (CRITICAL FIX) ----------------
@st.cache_resource
def load_chunks():
    with open("indexv2/chunks.pkl", "rb") as f:
        rulings_chunks = pickle.load(f)

    with open("indexv2/reg_chunks.pkl", "rb") as f:
        reg_chunks = pickle.load(f)

    return rulings_chunks, reg_chunks


# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    if file is None:
        return ""
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
    You are TNERC Legal Assistant, an intelligent legal copilot specializing in electricity regulations, consumer grievances, and Ombudsman proceedings.

    Your responsibilities depend on the user's request.

    GENERAL BEHAVIOR:
    - Behave conversationally and naturally like ChatGPT.
    - Understand the user's intent before responding.
    - Use uploaded documents, user input, URLs, regulations, and past rulings as context.
    - Use retrieved regulations and precedents to improve legal accuracy.
    - Do not blindly generate formal judgments unless requested.

    DOCUMENT HANDLING:
    - If the user uploads documents and asks questions, answer strictly from the document when possible.
    - If information is not explicitly available in the document, clearly say so.
    - Do NOT hallucinate missing facts, clauses, dates, or evidence.
    - You may provide a best-effort draft or suggestion, but clearly mention that the user should verify it.

    LEGAL ASSISTANCE:
    Depending on the request, you may:
    - summarize documents
    - explain legal language
    - rewrite content formally
    - draft complaints/replies
    - generate legal notices
    - generate Ombudsman-style orders
    - analyze applicability of regulations
    - compare with past rulings
    - explain regulations conversationally

    JUDGMENT GENERATION:
    Generate a full structured legal order ONLY if:
    - the user explicitly requests it
    OR
    - sufficient facts and supporting context are available.

    When generating judgments, use sections like:
    - Facts of the Case
    - Issues for Determination
    - Analysis
    - Findings
    - Relevant Regulations
    - Supporting Precedents
    - Final Order

    CONVERSATIONAL MODE:
    If the user provides incomplete information:
    - ask intelligent follow-up questions
    - guide them step-by-step
    - help them build the case progressively

    REFERENCE MATERIALS:
    Relevant Regulations:
    {''.join([r["text"][:500] for r in regs])}

    Relevant Past Rulings:
    {''.join([r["text"][:500] for r in rulings])}

    Current User Request:
    {query}
    """
    recent_history = st.session_state.messages[-6:]

    messages = [
    {
        "role": "system",
        "content": prompt
    }
    ]
    messages.extend(recent_history)
    messages.append({
    "role": "user",
    "content": prompt
    })

    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    st.session_state.messages.append({
    "role": "assistant",
    "content": response.choices[0].message.content
    })

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_text = st.chat_input("Describe the case or ask a legal question...")
    
    if user_text:
        st.session_state.messages.append({
        "role": "user",
        "content": user_text
        })  

    #uploaded_file = st.file_uploader("Upload case file", type=["pdf", "docx"])
    with st.sidebar:
        uploaded_file = st.file_uploader(
        "Upload case file",
        type=["pdf", "docx"]
        )
    file_text = ""
    if uploaded_file is not None:
        file_text = extract_text(uploaded_file)
    #if uploaded_file is None:
        #file_text = ""
    combined_text = f"""
    User Input:
    {user_text}
    Document Content:
    {file_text}
    """
    query = combined_text#[:6000]
    if not query.strip():
        st.warning("Please upload a file or enter case details")
        st.stop()
    if user_text:

        with st.spinner("Analyzing..."):
            rulings, regs = search_all(query, rulings_chunks, reg_chunks)
            answer = generate_answer(query, rulings, regs)

        with st.chat_message("assistant"):
            st.markdown(answer)

        if regs:
            st.subheader("Relevant Regulations")
            for r in regs:
                st.write(r["text"][:500])

        if rulings:
            st.subheader("Similar Past Rulings")
            for r in rulings:
                st.write(r["text"][:500])
