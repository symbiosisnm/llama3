"""
Email and sales assistant script for business LLMs.

This script demonstrates how to integrate a fine -tuned language model with business email
workflows. It connects to an IMAP server to retrieve unread emails, retrieves relevant
business context from a vector database using sentence embeddings, generates a reply using
the language model, and sends the reply via SMTP. Environment variables configure the
email servers, model paths, and vector database.

NOTE: This script is provided as a template. It assumes that IMAP and SMTP credentials
are available via environment variables. In production, handle credentials securely and
add proper error handling, logging, and message deduplication.

"""

import os
import imaplib
import smtplib
import email
from email.message import EmailMessage
from email.header import decode_header
from typing import List, Tuple

import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb



def load_text_generator(model_path: str):
    """Load a fine -tuned causal language model and return a text generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def get_vector_collection(chroma_dir: str, collection_name: str):
    """Return a ChromaDB collection for retrieving business context."""
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(name=collection_name)


def retrieve_context(collection, embedder, query: str, top_k: int = 3) -> str:
    """Retrieve top‑k relevant documents for a query and concatenate their text."""
    query_embedding = embedder.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "\n".join(docs)


def fetch_unseen_messages(imap_host: str, username: str, password: str, folder: str = "INBOX") -> List[Tuple[str, str, str]]:
    """
    Fetch unseen messages from the IMAP server.

    Returns a list of tuples (msg_id, subject, body).
    """
    conn = imaplib.IMAP4_SSL(imap_host)
    conn.login(username, password)
    conn.select(folder)
    typ, data = conn.search(None, "(UNSEEN)")
    messages: List[Tuple[str, str, str]] = []
    for msg_num in data[0].split():
        typ, msg_data = conn.fetch(msg_num, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        # Decode subject
        decoded_subject, encoding = decode_header(msg.get("Subject"))[0]
        if isinstance(decoded_subject, bytes):
            subject = decoded_subject.decode(encoding or "utf-8", errors="ignore")
        else:
            subject = decoded_subject or ""
        # Extract plain text body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in disposition:
                    charset = part.get_content_charset() or "utf-8"
                    body_bytes = part.get_payload(decode=True)
                    if body_bytes:
                        body = body_bytes.decode(charset, errors="ignore")
                        break
        else:
            charset = msg.get_content_charset() or "utf-8"
            body_bytes = msg.get_payload(decode=True)
            if body_bytes:
                body = body_bytes.decode(charset, errors="ignore")
        messages.append((msg_num.decode(), subject, body))
    conn.logout()
    return messages


def send_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    to_address: str,
    subject: str,
    body: str,
):
    """Send an email via SMTP."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = to_address
    msg.set_content(body)
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
        smtp.login(username, password)
        smtp.send_message(msg)


def generate_reply(generator, context: str, user_message: str, max_new_tokens: int = 256) -> str:
    """Generate a reply given context and user message."""
    prompt = f"{context}\n\nUser: {user_message}\nAssistant:"
    outputs = generator(prompt, max_new_tokens=max_new_tokens)
    generated = outputs[0]["generated_text"]
    # Return only the assistant's part after the last "Assistant:" token
    if "Assistant:" in generated:
        return generated.split("Assistant:")[-1].strip()
    return generated.strip()


def process_inbox():
    """Main entrypoint to process unread emails and generate replies."""
    # Environment configuration
    imap_host = os.environ.get("IMAP_HOST")
    imap_user = os.environ.get("IMAP_USER")
    imap_password = os.environ.get("IMAP_PASSWORD")
    smtp_host = os.environ.get("SMTP_HOST", imap_host)
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user = os.environ.get("SMTP_USER", imap_user)
    smtp_password = os.environ.get("SMTP_PASSWORD", imap_password)
    model_path = os.environ.get("MODEL_PATH", "./business-llama3")
    chroma_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
    collection_name = os.environ.get("COLLECTION_NAME", "business_docs")
    embed_model_name = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    # Load LLM and embeddings
    generator = load_text_generator(model_path)
    embedder = SentenceTransformer(embed_model_name)
    collection = get_vector_collection(chroma_dir, collection_name)
    # Fetch unread emails
    messages = fetch_unseen_messages(imap_host, imap_user, imap_password)
    for msg_id, subject, body in messages:
        user_query = f"Subject: {subject}\n\n{body}"
        context = retrieve_context(collection, embedder, user_query, top_k=3)
        reply = generate_reply(generator, context, user_query)
        # Send reply to yourself or to the original sender. Here we send to self for demonstration.
        try:
            send_email(smtp_host, smtp_port, smtp_user, smtp_password, imap_user, f"Re: {subject}", reply)
            logging.info("Replied to message %s", msg_id)
        except Exception as e:
            logging.error("Failed to send reply for %s: %s", msg_id, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_inbox()
