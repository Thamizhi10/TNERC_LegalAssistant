import streamlit as st
import faiss
import pickle
import numpy as np
import os
import zipfile
import requests
import fitz
import docx

from openai import OpenAI

# ---------------- CONFIG ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_ZIP_URL = "https://drive.google.com/uc?id=1MUvU7ByNbVpxE4COU_rCXlqvkUFDevnA"

# ---------------- DOWNLOAD INDEX ----------------
def download_and_extract():
    if not os.path.exists("index"):
        st.write("Downloading knowledge base...")

        r = requests.get(INDEX_ZIP_URL)
        with open("index.zip", "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile("index.zip", 'r') as zip_ref:
            zip_ref.extractall()

        st.write("Ready")

# ---------------- LOAD INDEXES ----------------
def load_indexes():
    rulings_index = faiss.read_index("index/rulings.index")

    with open("index/chunks.pkl", "rb") as f:
        rulings_chunks = pickle.load(f)

    reg_index = faiss.read_index("index/regulations.index")

    with open("index/reg_chunks.pkl", "rb") as f:
        reg_chunks = pickle.load(f)

    return rulings_index, rulings_chunks, reg_index, reg_chunks

# ---------------- EXTRACT TEXT ----------------
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

# ---------------- EMBEDDING ----------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ---------------- SEARCH ----------------
def search_all(query, rulings_index, rulings_chunks, reg_index, reg_chunks):
    q_emb = get_embedding(query)

    # Rulings
    D1, I1 = rulings_index.search(np.array([q_emb]).astype("float32"), k=3)
    rulings = [rulings_chunks[i] for i in I1[0]]

    # Regulations
    D2, I2 = reg_index.search(np.array([q_emb]).astype("float32"), k=3)
    regs = [reg_chunks[i] for i in I2[0]]

    return rulings, regs

# ---------------- UI ----------------
st.title("TNERC Legal Assistant")

download_and_extract()

rulings_index, rulings_chunks, reg_index, reg_chunks = load_indexes()

uploaded_file = st.file_uploader("Upload case file", type=["pdf", "docx"])

if uploaded_file is not None:

    text = extract_text(uploaded_file)
    query = text[:2000]

    if st.button("Analyze Case"):

        rulings, regs = search_all(query, rulings_index, rulings_chunks, reg_index, reg_chunks)

        st.subheader("Relevant Regulations")
        for r in regs:
            st.write(r["text"][:500])

        st.subheader("Similar Past Rulings")
        for r in rulings:
            st.write(r["text"][:500])