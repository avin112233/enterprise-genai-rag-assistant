
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("enterprise_knowledge")


def retrieve_documents(question, top_k=3):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    return [
        {
            "text": document,
            "distance": distance,
            "metadata": metadata,
        }
        for document, distance, metadata in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]


if __name__ == "__main__":
    questions = [
        "What validations are performed for Contact BP?",
        "Explain Hexagon scoring.",
        "How do I troubleshoot a QA failure?",
    ]

    for question in questions:
        print("\n" + "=" * 65)
        print("QUESTION:", question)
        print("=" * 65)

        results = retrieve_documents(question, top_k=2)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Chunk ID: {result['metadata']['chunk_id']}")
            print(f"Distance: {result['distance']:.4f}")
            print("Document:")
            print(result["text"])
