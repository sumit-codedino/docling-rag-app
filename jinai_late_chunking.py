from langchain.text_splitter import MarkdownTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_markdown_embeddings(
    file_path: str,
    model_name: str = 'jinaai/jina-embeddings-v2-base-en',
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    device: str = None,
    index_name: str = "markdown_embeddings"
) -> Dict[str, Any]:
    """Generate embeddings from a markdown file using Jina AI's late chunking technique.
    
    Args:
        file_path: Path to markdown file
        model_name: Name of the embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        device: Device to use (cuda/cpu)
        index_name: OpenSearch index name
        
    Returns:
        Dictionary containing documents ready for OpenSearch injection
    """
    try:
        # Set device
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        
        # Load markdown content
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.suffix.lower() == '.md':
            raise ValueError(f"Invalid file type: {file_path.suffix}. Only .md files are supported.")
            
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
            
        if not content.strip():
            raise ValueError("File is empty")
            
        # LATE CHUNKING IMPLEMENTATION
        logger.info("Implementing late chunking technique")
        
        # Step 1: Tokenize the entire document
        logger.info("Tokenizing entire document")
        tokenization_output = tokenizer(
            content,
            return_offsets_mapping=True,
            return_tensors='pt',
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False
        )
        
        # Step 2: Get embeddings for all tokens
        logger.info("Generating token-level embeddings")
        model_inputs = {
            'input_ids': tokenization_output['input_ids'].to(device),
            'attention_mask': tokenization_output['attention_mask'].to(device)
        }
        
        with torch.no_grad():
            outputs = model(**model_inputs)
            # Get token-level embeddings (not pooled)
            token_embeddings = outputs.last_hidden_state.squeeze()  # [seq_len, hidden_dim]
            
        offset_mapping = tokenization_output['offset_mapping'].squeeze().tolist()
        
        # Step 3: Create text chunks AFTER getting embeddings
        logger.info("Creating markdown chunks")
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = splitter.split_text(content)
        
        # Step 4: Apply late chunking - map chunks to token embeddings
        opensearch_docs = []
        
        for i, chunk_text in enumerate(text_chunks):
            # Find character positions of this chunk in original text
            start_char = content.find(chunk_text)
            if start_char == -1:
                # Fallback: search more flexibly
                chunk_words = chunk_text.split()[:5]  # First 5 words
                search_text = ' '.join(chunk_words)
                start_char = content.find(search_text)
                if start_char == -1:
                    start_char = 0
            
            end_char = start_char + len(chunk_text)
            
            # Map character positions to token positions
            chunk_token_indices = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                # Check if token overlaps with chunk
                if (token_start < end_char and token_end > start_char):
                    chunk_token_indices.append(token_idx)
            
            # Handle edge case where no tokens found
            if not chunk_token_indices:
                chunk_token_indices = [0]
            
            # Apply chunk-aware pooling (mean pooling for Jina models)
            chunk_token_embeddings = token_embeddings[chunk_token_indices]
            
            # Late chunking: weighted average based on token relevance
            if len(chunk_token_indices) > 1:
                # Use attention-like weighting for better representation
                weights = torch.softmax(torch.norm(chunk_token_embeddings, dim=1), dim=0)
                chunk_embedding = torch.sum(chunk_token_embeddings * weights.unsqueeze(1), dim=0)
            else:
                chunk_embedding = chunk_token_embeddings[0]
            
            # Convert to numpy for JSON serialization
            chunk_embedding_np = chunk_embedding.cpu().numpy()
            
            # Generate chunk ID
            content_hash = hashlib.md5(
                f"{chunk_text}_{str(file_path)}_{i}".encode()
            ).hexdigest()[:12]
            chunk_id = f"{file_path.stem}_chunk_{i:03d}_{content_hash}"
            
            # Create OpenSearch document
            opensearch_doc = {
                "_index": index_name,
                "_id": chunk_id,
                "_source": {
                    "text": chunk_text.strip(),
                    "embedding": chunk_embedding_np.tolist(),
                    "source_file": str(file_path),
                    "file_name": file_path.name,
                    "chunk_index": i,
                    "char_start": start_char,
                    "char_end": end_char,
                    "token_count": len(chunk_token_indices),
                    "created_at": datetime.now().isoformat(),
                    "model_used": model_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            }
            
            opensearch_docs.append(opensearch_doc)
        
        # Prepare final result
        result = {
            "success": True,
            "documents": opensearch_docs,  # Ready for OpenSearch bulk API
            "processing_info": {
                "total_chunks": len(opensearch_docs),
                "embedding_dimension": len(opensearch_docs[0]["_source"]["embedding"]) if opensearch_docs else 0,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model_used": model_name,
                "device_used": device,
                "source_file": str(file_path),
                "index_name": index_name,
                "late_chunking": True,
                "total_tokens": len(offset_mapping)
            }
        }
        
        logger.info(f"Successfully generated {len(opensearch_docs)} chunks using late chunking technique")
        return result
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "file_path": str(file_path),
            "documents": [],
            "processing_info": {
                "late_chunking": False,
                "error_occurred": True
            }
        }

# Example usage
if __name__ == "__main__":
    result = generate_markdown_embeddings(
        "output/emami.md",
        chunk_size=512,
        chunk_overlap=50,
        index_name="my_markdown_index"
    )
    
    if result["success"]:
        print(f"Generated {len(result['documents'])} documents ready for OpenSearch")
        print(f"Embedding dimension: {result['processing_info']['embedding_dimension']}")
        
        # The result['documents'] can now be directly used with OpenSearch bulk API
        # Each document in the format: {"_index": "...", "_id": "...", "_source": {...}}
    else:
        print(f"Error: {result['error']}")