from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import tempfile
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Docling + Vector Search API", version="0.3")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq API
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please set it as an environment variable.")

# Initialize Sentence Transformer Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "pdf_collection"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# ✅ Clear ChromaDB after query is completed
def clear_chroma_collection():
    """Clear the entire ChromaDB collection."""
    chroma_collection.delete(ids=chroma_collection.get()['ids'])
    logger.info("ChromaDB collection cleared successfully.")


# ✅ Convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path: Path):
    """Convert PDF to Markdown using Docling."""
    logger.info(f"Processing PDF: {pdf_path}")
    converter = DocumentConverter()
    result = converter.convert(source=pdf_path)

    # Extract markdown text from the document
    md_text = result.document.export_to_markdown()
    logger.info("Document converted to markdown successfully.")
    return md_text


# ✅ Chunk Markdown for Embedding
def chunk_document(document_text, chunk_size=500):
    """Split document into smaller chunks."""
    words = document_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info(f"Document chunked into {len(chunks)} parts.")
    return chunks


# ✅ Embed and Store Chunks in ChromaDB
def store_chunks_in_chroma(chunks, file_name):
    """Generate embeddings and store in ChromaDB."""
    embeddings = embedder.encode(chunks)
    documents = [{"id": f"{file_name}_{i}", "content": chunk} for i, chunk in enumerate(chunks)]
    for i, embedding in enumerate(embeddings):
        chroma_collection.add(
            ids=[documents[i]["id"]],
            embeddings=[embedding.tolist()],
            metadatas=[{"filename": file_name, "chunk_index": i}],
            documents=[documents[i]["content"]]
        )
    logger.info(f"Stored {len(chunks)} document chunks in ChromaDB.")


# ✅ Query ChromaDB for Relevant Chunks
def query_chroma(query_text, top_k=3):
    """Retrieve top-k relevant chunks from ChromaDB."""
    query_embedding = embedder.encode(query_text).tolist()
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    # Combine top results as context for Groq
    if results["documents"]:
        combined_context = "\n\n".join(results["documents"][0])
        logger.info(f"Retrieved {len(results['documents'][0])} relevant chunks from ChromaDB.")
        return combined_context
    else:
        logger.warning("No relevant chunks found.")
        return "No relevant information found."


# ✅ Query Groq with Relevant Content
def query_groq(context):
    """Send retrieved chunks and query to Groq."""
    prompt = f"""
    You are an intelligent document assistant. Here is the content of the document:

    {context}

    
    Please extract first name, last name,DL number and address and return the requested information **strictly** in valid JSON format without any additional text. 
    If any of the requested fields are not found, set their value to null.

    eg. {{"first_name": "John", "last_name": "Doe", "address": "123 Main St, Springfield, IL 62701"}}
    """

    try:
        # Use the Groq client to send the query
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
        )
        
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content
        else:
            return "No valid response received from Groq."

    except Exception as e:
        logger.error(f"Error communicating with Groq library: {str(e)}")
        raise HTTPException(status_code=500, detail="Error communicating with Groq library.")


# 🎯 Single API for PDF Upload, Query, and Retrieval
@app.post("/query")
async def process_and_query(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """Upload PDF, generate embeddings, store in ChromaDB, and query."""
    logger.info(f"Received query request with file: {file.filename} and query: {query}")

    if not file.filename.lower().endswith(".pdf"):
        logger.error("Invalid file format. Only PDF files are allowed.")
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    try:
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        logger.info(f"File saved temporarily at {temp_file_path}")
        temp_file_path_obj = Path(temp_file_path)

        # Convert to Markdown and Chunk
        document_text = convert_pdf_to_markdown(temp_file_path_obj)
        chunks = chunk_document(document_text)

        # Store Chunks in ChromaDB
        store_chunks_in_chroma(chunks, file.filename)

        # Query ChromaDB for Relevant Chunks
        relevant_context = query_chroma(query)

        # Query Groq with Relevant Context
        groq_response = query_groq(relevant_context)

        clear_chroma_collection()

        # Clean up temp file after processing
        os.remove(temp_file_path)
        logger.info(f"Temporary file {temp_file_path} deleted after processing.")

        return {
            "filename": file.filename,
            "query": query,
            "response": groq_response,
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))