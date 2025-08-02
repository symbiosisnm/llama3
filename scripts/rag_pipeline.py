import os
import argparse

from typing import List

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def ingest_documents(docs_dir: str, collection_name: str = "business_docs", persist_directory: str = "./chroma_db"):
    """
    Read text files from `docs_dir`, embed them with SentenceTransformer, and store in a ChromaDB collection.
    Each file is treated as a single document.
    """
    # Initialize Chroma client with persistent storage
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    doc_texts: List[str] = []
    embeddings: List[list] = []
    ids: List[str] = []

    for root, _, files in os.walk(docs_dir):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # skip files that can't be decoded
                continue
            doc_id = os.path.relpath(path, docs_dir)
            embedding = model.encode(text)
            doc_texts.append(text)
            embeddings.append(embedding)
            ids.append(doc_id)

    if doc_texts:
        collection.add(documents=doc_texts, embeddings=embeddings, ids=ids)
        client.persist()
        print(f"Ingested {len(ids)} documents into collection '{collection_name}'.")
    else:
        print("No text documents found to ingest.")


def query_documents(query: str, collection_name: str = "business_docs", top_k: int = 3, persist_directory: str = "./chroma_db"):
    """
    Query the ChromaDB collection for relevant documents given a query string.
    Returns the top_k documents and their distances.
    """
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    collection = client.get_collection(name=collection_name)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results['documents'][0]
    ids = results['ids'][0]
    distances = results['distances'][0]
    return docs, ids, distances


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into a Chroma collection.")
    ingest_parser.add_argument("--docs_dir", type=str, required=True, help="Directory containing text documents.")
    ingest_parser.add_argument("--collection_name", type=str, default="business_docs", help="Name of the Chroma collection.")
    ingest_parser.add_argument("--persist_directory", type=str, default="./chroma_db", help="Directory to store persistent Chroma DB.")

    query_parser = subparsers.add_parser("query", help="Query a Chroma collection for relevant documents.")
    query_parser.add_argument("--query", type=str, required=True, help="User query text.")
    query_parser.add_argument("--collection_name", type=str, default="business_docs", help="Name of the Chroma collection.")
    query_parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve.")
    query_parser.add_argument("--persist_directory", type=str, default="./chroma_db", help="Directory of persistent Chroma DB.")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_documents(args.docs_dir, args.collection_name, args.persist_directory)
    elif args.command == "query":
        docs, ids, distances = query_documents(args.query, args.collection_name, args.top_k, args.persist_directory)
        for doc, doc_id, dist in zip(docs, ids, distances):
            print(f"ID: {doc_id}")
            print(f"Distance: {dist:.4f}")
            print(f"Content snippet: {doc[:300]}")
            print("-" * 40)


if __name__ == "__main__":
    main()
