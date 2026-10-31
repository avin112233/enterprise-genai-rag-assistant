
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="enterprise_knowledge"
)

text = Path("data/knowledge.txt").read_text(
    encoding="utf-8"
)

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

embeddings = model.encode(chunks).tolist()

collection.upsert(
    ids=[f"doc_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {"source": "knowledge.txt", "chunk_id": i}
        for i in range(len(chunks))
    ],
)

print(f"Successfully indexed {len(chunks)} document chunks")
print("ChromaDB vector database is ready")
