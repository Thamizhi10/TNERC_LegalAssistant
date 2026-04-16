import faiss
import pickle
import numpy as np

# =========================
# PROCESS RULINGS
# =========================

print("Processing rulings...")

index = faiss.read_index("index/rulings.index")
vectors = index.reconstruct_n(0, index.ntotal)

with open("index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

valid_chunks = [c for c in chunks if c.get("text", "").strip()]

print(f"Chunks: {len(chunks)}")
print(f"Valid chunks: {len(valid_chunks)}")
print(f"Vectors: {len(vectors)}")

for chunk, emb in zip(valid_chunks, vectors):
    chunk["embedding"] = np.array(emb, dtype=np.float16).tolist()  # convert to list for safe storage

with open("index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Rulings embeddings attached")


# =========================
# PROCESS REGULATIONS
# =========================

print("\nProcessing regulations...")

index = faiss.read_index("index/regulations.index")
vectors = index.reconstruct_n(0, index.ntotal)

with open("index/reg_chunks.pkl", "rb") as f:
    reg_chunks = pickle.load(f)

valid_chunks = [c for c in reg_chunks if c.get("text", "").strip()]

print(f"Chunks: {len(reg_chunks)}")
print(f"Valid chunks: {len(valid_chunks)}")
print(f"Vectors: {len(vectors)}")

for chunk, emb in zip(valid_chunks, vectors):
    chunk["embedding"] = np.array(emb, dtype=np.float16).tolist()

with open("index/reg_chunks.pkl", "wb") as f:
    pickle.dump(reg_chunks, f)

print("Regulations embeddings attached")

print("\nDONE")