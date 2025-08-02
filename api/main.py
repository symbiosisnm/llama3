import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch


class GenerateRequest(BaseModel):
    prompt: str
    top_k: int = 3
    max_new_tokens: int = 256


class GenerateResponse(BaseModel):
    response: str


app = FastAPI(title="Business Assistant API", version="1.0.0")

# Configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "./business-llama3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "business_docs")

# Load generation model and tokenizer
print(f"Loading generation model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load embedding model and vector database
print("Loading embedding model and vector database...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)


def get_context(query: str, top_k: int) -> str:
    """
    Retrieve top_k documents from the ChromaDB collection based on similarity to the query.
    Returns a single string containing the concatenated documents.
    """
    query_embedding = embedder.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results["documents"][0]
    return "\n".join(docs)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate a response for the given user prompt, using retrieval‑augmented generation.
    """
    # Retrieve relevant context from the vector store
    context = get_context(req.prompt, req.top_k)
    # Construct the combined prompt for the LLM
    combined_prompt = f"{context}\n\nUser: {req.prompt}\nAssistant:"
    # Encode the prompt and generate
    inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode and extract assistant response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Assistant:" in generated_text:
        # Take text after the last 'Assistant:'
        response = generated_text.split("Assistant:")[-1].strip()
    else:
        response = generated_text
    return GenerateResponse(response=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
