import os
import faiss
import pickle
import numpy as np

from ingestion import ingest_subject, ingest_regulations
from embeddings import get_embedding
from logger import get_logger
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- PATHS ----------------
RULINGS_INDEX_PATH = "index/rulings.index"
RULINGS_CHUNKS_PATH = "index/chunks.pkl"

REG_INDEX_PATH = "index/regulations.index"
REG_CHUNKS_PATH = "index/reg_chunks.pkl"

logger = get_logger()


if __name__ == "__main__":
    
    subject = "HC MADRAS Meter Defects" 

    logger.info("===== STARTING PIPELINE =====")

    os.makedirs("index", exist_ok=True)

    # =========================================================
    # STEP 1: REGULATIONS (RUN ONLY ONCE)
    # =========================================================

    if os.path.exists(REG_INDEX_PATH) and os.path.exists(REG_CHUNKS_PATH):
        logger.info("Regulations index already exists")
    else:
        logger.info("Building regulations index...")

        reg_chunks = ingest_regulations()

        texts = [c["text"] for c in reg_chunks if c["text"].strip()]
        embeddings = [get_embedding(t) for t in texts]


        dimension = len(embeddings[0])
        reg_index = faiss.IndexFlatL2(dimension)
        reg_index.add(np.array(embeddings).astype("float32"))

        faiss.write_index(reg_index, REG_INDEX_PATH)

        with open(REG_CHUNKS_PATH, "wb") as f:
            pickle.dump(reg_chunks, f)

        logger.info(f"Regulations indexed: {len(reg_chunks)} chunks")

    # =========================================================
    # STEP 2: RULINGS (INCREMENTAL)
    # =========================================================

    if os.path.exists(RULINGS_INDEX_PATH) and os.path.exists(RULINGS_CHUNKS_PATH):
        index = faiss.read_index(RULINGS_INDEX_PATH)

        with open(RULINGS_CHUNKS_PATH, "rb") as f:
            all_chunks = pickle.load(f)

        logger.info("Loaded rulings index")
    else:
        index = None
        all_chunks = []
        logger.info("Creating new rulings index")

    # -------- PROCESS SUBJECT --------
    new_chunks = ingest_subject(subject)
    all_chunks.extend(new_chunks)

    logger.info(f"New chunks: {len(new_chunks)}")

    #texts = [c["text"] for c in new_chunks if c["text"].strip()]
    #embeddings = [get_embedding(t) for t in texts]
    texts = [c["text"] for c in new_chunks if c["text"].strip()]

    '''response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
    )

    embeddings = [d.embedding for d in response.data]    '''
    embeddings = []

    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)

        logger.info(f"Processed batch {i} to {i+len(batch)}")

    if index is None:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))

    logger.info(f"Rulings index size: {index.ntotal}")

    # -------- SAVE --------
    faiss.write_index(index, RULINGS_INDEX_PATH)

    with open(RULINGS_CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Saved subject: {subject}")

    logger.info("===== PROCESS COMPLETED =====")