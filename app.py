import streamlit as st
import pickle
import numpy as np
import os
import zipfile
import requests
import fitz
import docx

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_ZIP_URL = "https://drive.google.com/uc?id=1MUvU7ByNbVpxE4COU_rCXlqvkUFDevnA"

def download_and_extract():
    if not os.path.exists("index"):
        st.write("Downloading knowledge base...")

        r = requests.get(INDEX_ZIP_URL)

        if r.status_code != 200:
            st.error("Failed to download index. Check Google Drive link.")
            st.stop()

        with open("index.zip", "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile("index.zip", 'r') as zip_ref:
            zip_ref.extractall()

        st.write("Ready")

def load_chunks():
    with open("index/chunks.pkl", "rb") as f:
        rulings_chunks = pickle.load(f)

    with open("index/reg_chunks.pkl", "rb") as f:
        reg_chunks = pickle.load(f)

    return rulings_chunks, reg_chunks

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

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def simple_search(query_emb, chunks, top_k=3):
    scores = []

    for c in chunks:
        if "embedding" not in c:
            continue

        emb = np.array(c["embedding"])
        score = np.dot(query_emb, emb)
        scores.append((score, c))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scores[:top_k]]

def search_all(query, rulings_chunks, reg_chunks):
    q_emb = np.array(get_embedding(query))

    rulings = simple_search(q_emb, rulings_chunks)
    regs = simple_search(q_emb, reg_chunks)

    return rulings, regs

st.title("TNERC Legal Assistant")

download_and_extract()

rulings_chunks, reg_chunks = load_chunks()

uploaded_file = st.file_uploader("Upload case file", type=["pdf", "docx"])

if uploaded_file is not None:

    text = extract_text(uploaded_file)
    query = text[:2000]

    if st.button("Analyze Case"):

        rulings, regs = search_all(query, rulings_chunks, reg_chunks)

        st.subheader("Relevant Regulations")
        for r in regs:
            st.write(r["text"][:500])

        st.subheader("Similar Past Rulings")
        for r in rulings:
            st.write(r["text"][:500])