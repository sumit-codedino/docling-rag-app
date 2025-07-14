from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, PrivateAttr, Field
from pathlib import Path
import nest_asyncio
from dotenv import load_dotenv
import os
import asyncio
from logger import setup_logger
from typing import Optional, Dict, Any, List, Tuple
import json
import sys
from datetime import datetime
from transformers import AutoModel, AutoTokenizer, BatchEncoding
import requests
import torch
import numpy as np
import math

# Load environment variables
load_dotenv()
nest_asyncio.apply()
logger = setup_logger()

class JinaChunkingTool(BaseModel):
    """Tool for embedding and storing markdown files using Jina embeddings with late chunking"""
    
    _logger: Any = PrivateAttr(default=None)
    _tokenizer: Optional[AutoTokenizer] = PrivateAttr(default=None)
    _model: Optional[AutoModel] = PrivateAttr(default=None)
    _jina_max_chars: int = PrivateAttr(default=64000)  # Jina API limit
    
    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logger
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Jina model and tokenizer"""
        try:
            self._logger.info("Loading Jina model and tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                'jinaai/jina-embeddings-v2-base-en', 
                trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                'jinaai/jina-embeddings-v2-base-en', 
                trust_remote_code=True
            )
            self._logger.info("Successfully loaded Jina model and tokenizer")
        except Exception as e:
            self._logger.error(f"Error loading Jina model: {str(e)}")
            raise

    async def load_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Load markdown file content
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dict containing file content and metadata
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self._logger.error(f"Markdown file not found: {file_path_obj}")
                raise FileNotFoundError(f"Markdown file not found: {file_path_obj}")

            self._logger.info(f"Loading markdown file: {file_path_obj}")
            content = file_path_obj.read_text(encoding='utf-8')
            
            metadata = {
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj.absolute()),
                "file_size": file_path_obj.stat().st_size,
                "last_modified": datetime.fromtimestamp(file_path_obj.stat().st_mtime).isoformat(),
                "encoding": "utf-8",
                "content_length": len(content),
                "exceeds_jina_limit": len(content) > self._jina_max_chars
            }
            
            self._logger.info(f"Successfully loaded markdown file: {file_path_obj} ({len(content)} chars)")
            if len(content) > self._jina_max_chars:
                self._logger.warning(f"File exceeds Jina API limit ({len(content)} > {self._jina_max_chars} chars)")
            
            return {
                "success": True,
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            self._logger.error(f"Error loading markdown file {file_path}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "source_file": file_path
            }

    def _split_text_by_sections(self, text: str, max_chars: int) -> List[str]:
        """Split text into sections that respect markdown structure and character limits
        
        Args:
            text: Input text to split
            max_chars: Maximum characters per section
            
        Returns:
            List of text sections
        """
        if len(text) <= max_chars:
            return [text]
        
        sections = []
        
        # Try to split by markdown headers first
        lines = text.split('\n')
        current_section = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed the limit
            if current_length + line_length > max_chars and current_section:
                # Save current section
                sections.append('\n'.join(current_section))
                current_section = [line]
                current_length = line_length
            else:
                current_section.append(line)
                current_length += line_length
        
        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If any section is still too large, split it further
        final_sections = []
        for section in sections:
            if len(section) > max_chars:
                # Split large sections by paragraphs or sentences
                final_sections.extend(self._split_large_section(section, max_chars))
            else:
                final_sections.append(section)
        
        return final_sections

    def _split_large_section(self, text: str, max_chars: int) -> List[str]:
        """Split a large section into smaller parts
        
        Args:
            text: Text to split
            max_chars: Maximum characters per part
            
        Returns:
            List of text parts
        """
        if len(text) <= max_chars:
            return [text]
        
        parts = []
        
        # Try splitting by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        current_part = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph) + 2  # +2 for double newline
            
            if current_length + paragraph_length > max_chars and current_part:
                parts.append('\n\n'.join(current_part))
                current_part = [paragraph]
                current_length = len(paragraph)
            else:
                current_part.append(paragraph)
                current_length += paragraph_length
        
        if current_part:
            parts.append('\n\n'.join(current_part))
        
        # If still too large, split by sentences
        final_parts = []
        for part in parts:
            if len(part) > max_chars:
                sentences = part.replace('.', '.\n').split('\n')
                current_sentence_group = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_length = len(sentence) + 1
                    
                    if current_length + sentence_length > max_chars and current_sentence_group:
                        final_parts.append(' '.join(current_sentence_group))
                        current_sentence_group = [sentence]
                        current_length = sentence_length
                    else:
                        current_sentence_group.append(sentence)
                        current_length += sentence_length
                
                if current_sentence_group:
                    final_parts.append(' '.join(current_sentence_group))
            else:
                final_parts.append(part)
        
        return final_parts

    async def chunk_by_jina_api(self, input_text: str, max_chunk_length: int = 1000) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Create chunks using Jina's segmentation API with 64k character limit handling
        
        Args:
            input_text: Text to be chunked
            max_chunk_length: Maximum length of each chunk
            
        Returns:
            Tuple of (chunks, span_annotations)
        """
        try:
            all_chunks = []
            all_span_annotations = []
            
            # If text is within Jina's limit, process directly
            if len(input_text) <= self._jina_max_chars:
                return await self._process_with_jina_api(input_text, max_chunk_length)
            
            # Split text into sections that fit within Jina's limit
            self._logger.info(f"Text too large ({len(input_text)} chars), splitting into sections...")
            sections = self._split_text_by_sections(input_text, self._jina_max_chars - 1000)  # Leave some buffer
            
            current_offset = 0
            
            for i, section in enumerate(sections):
                self._logger.info(f"Processing section {i+1}/{len(sections)} ({len(section)} chars)")
                
                try:
                    section_chunks, section_spans = await self._process_with_jina_api(section, max_chunk_length)
                    
                    # Adjust span annotations to account for the offset in the full text
                    adjusted_spans = [(start + current_offset, end + current_offset) for start, end in section_spans]
                    
                    all_chunks.extend(section_chunks)
                    all_span_annotations.extend(adjusted_spans)
                    
                    # Update offset for next section
                    current_offset += len(section)
                    
                except Exception as e:
                    self._logger.warning(f"Failed to process section {i+1} with Jina API: {str(e)}")
                    # Fallback to simple chunking for this section
                    section_chunks, section_spans = await self._fallback_chunking(section, max_chunk_length)
                    adjusted_spans = [(start + current_offset, end + current_offset) for start, end in section_spans]
                    all_chunks.extend(section_chunks)
                    all_span_annotations.extend(adjusted_spans)
                    current_offset += len(section)
            
            self._logger.info(f"Successfully processed large text into {len(all_chunks)} chunks across {len(sections)} sections")
            return all_chunks, all_span_annotations
            
        except Exception as e:
            self._logger.error(f"Error in Jina API chunking: {str(e)}")
            # Complete fallback to simple chunking
            return await self._fallback_chunking(input_text, max_chunk_length)

    async def _process_with_jina_api(self, text: str, max_chunk_length: int) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Process a single text section with Jina API
        
        Args:
            text: Text to process (must be <= 64k chars)
            max_chunk_length: Maximum chunk length
            
        Returns:
            Tuple of (chunks, span_annotations)
        """
        url = 'https://api.jina.ai/v1/segment'
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Add authorization if API key is available
        if os.getenv('JINA_API_KEY'):
            headers['Authorization'] = f'Bearer {os.getenv("JINA_API_KEY")}'
        
        payload = {
            "content": text,
            "return_chunks": True,
            "max_chunk_length": max_chunk_length
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Extract chunks and positions from the response
        chunks = response_data.get("chunks", [])
        chunk_positions = response_data.get("chunk_positions", [])

        # Convert to span annotations format
        span_annotations = [(start, end) for start, end in chunk_positions]
        
        return chunks, span_annotations

    async def _fallback_chunking(self, text: str, max_length: int = 1000) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Fallback chunking method if API fails"""
        chunks = []
        span_annotations = []
        
        words = text.split()
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                end_pos = start_pos + len(chunk_text)
                span_annotations.append((start_pos, end_pos))
                
                start_pos = end_pos + 1
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            end_pos = start_pos + len(chunk_text)
            span_annotations.append((start_pos, end_pos))
        
        return chunks, span_annotations

    async def late_chunking(self, model_output: torch.Tensor, span_annotations: List[Tuple[int, int]], max_length: Optional[int] = None) -> List[np.ndarray]:
        """Apply late chunking technique to get chunk embeddings
        
        Args:
            model_output: Token embeddings from the model
            span_annotations: List of (start, end) positions for each chunk
            max_length: Maximum sequence length to consider
            
        Returns:
            List of chunk embeddings
        """
        try:
            if max_length is not None:
                # Remove annotations that go beyond max_length
                span_annotations = [
                    (start, min(end, max_length - 1))
                    for start, end in span_annotations
                    if start < (max_length - 1)
                ]
            
            pooled_embeddings = []
            for start, end in span_annotations:
                if (end - start) >= 1:
                    # Average pooling over the token embeddings in the span
                    chunk_embedding = model_output[start:end].mean(dim=0)
                    pooled_embeddings.append(chunk_embedding.detach().cpu().numpy())
            
            self._logger.info(f"Created {len(pooled_embeddings)} chunk embeddings using late chunking")
            return pooled_embeddings
            
        except Exception as e:
            self._logger.error(f"Error in late chunking: {str(e)}")
            raise

    async def process_markdown_with_late_chunking(self, file_path: str, max_chunk_length: int = 1000) -> Dict[str, Any]:
        """Complete pipeline: load markdown, chunk, and create embeddings with late chunking
        
        Args:
            file_path: Path to markdown file
            max_chunk_length: Maximum length for each chunk
            
        Returns:
            Dict containing processed results
        """
        try:
            # Load markdown file
            file_result = await self.load_markdown_file(file_path)
            if not file_result["success"]:
                return file_result
            
            content = file_result["content"]
            metadata = file_result["metadata"]
            
            # Create chunks using Jina API (with 64k limit handling)
            chunks, span_annotations = await self.chunk_by_jina_api(content, max_chunk_length)
            
            # For large documents, we need to process embeddings in sections
            # since we can't tokenize the entire document at once
            if len(content) > self._jina_max_chars:
                processed_chunks = await self._process_large_document_embeddings(
                    content, chunks, span_annotations
                )
            else:
                processed_chunks = await self._process_small_document_embeddings(
                    content, chunks, span_annotations
                )
            
            # Prepare results
            result = {
                "success": True,
                "file_metadata": metadata,
                "processing_info": {
                    "total_chunks": len(chunks),
                    "chunking_method": "jina_late_chunking_with_64k_limit",
                    "max_chunk_length": max_chunk_length,
                    "model_used": "jinaai/jina-embeddings-v2-base-en",
                    "processed_at": datetime.now().isoformat(),
                    "document_split_into_sections": len(content) > self._jina_max_chars,
                    "jina_api_limit": self._jina_max_chars
                },
                "chunks": processed_chunks
            }
            
            self._logger.info(f"Successfully processed markdown file with {len(chunks)} chunks")
            return result
            
        except Exception as e:
            self._logger.error(f"Error in complete processing pipeline: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "source_file": file_path
            }

    async def _process_small_document_embeddings(self, content: str, chunks: List[str], span_annotations: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Process embeddings for documents that fit within token limits"""
        # Tokenize the full text for late chunking
        self._logger.info("Tokenizing full text for late chunking...")
        inputs = self._tokenizer(
            content,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8192  # Use larger context for late chunking
        )
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
        
        # Apply late chunking
        chunk_embeddings = await self.late_chunking(
            token_embeddings, 
            span_annotations,
            max_length=inputs['input_ids'].shape[1]
        )
        
        # Prepare processed chunks
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            processed_chunks.append({
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding.tolist(),
                "embedding_shape": embedding.shape,
                "span": span_annotations[i] if i < len(span_annotations) else None,
                "processing_method": "late_chunking"
            })
        
        return processed_chunks

    async def _process_large_document_embeddings(self, content: str, chunks: List[str], span_annotations: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Process embeddings for large documents that exceed token limits"""
        self._logger.info("Processing large document with individual chunk embeddings...")
        
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Create embedding for each chunk individually
            inputs = self._tokenizer(
                chunk,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling for chunk embedding
                chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                chunk_embedding_np = chunk_embedding.cpu().numpy()
            
            processed_chunks.append({
                "chunk_id": i,
                "text": chunk,
                "embedding": chunk_embedding_np.tolist(),
                "embedding_shape": chunk_embedding_np.shape,
                "span": span_annotations[i] if i < len(span_annotations) else None,
                "processing_method": "individual_chunk_embedding"
            })
        
        self._logger.info(f"Processed {len(processed_chunks)} chunks individually due to document size")
        return processed_chunks

# Define the agent context
class MarkdownProcessorContext(BaseModel):
    file_path: Optional[str] = None
    max_chunk_length: int = 1000
    output_file: Optional[str] = None

# Create the Pydantic AI Agent
markdown_agent = Agent(
    'openai:gpt-3.5-turbo',
    deps_type=JinaChunkingTool,
    result_type=Dict[str, Any],
    system_prompt="""
    You are a specialist in processing markdown files using Jina embeddings with late chunking technique.
    
    Your capabilities include:
    1. Loading markdown files from specified paths
    2. Handling large documents that exceed Jina's 64k character limit by splitting them intelligently
    3. Chunking text using Jina's segmentation API with proper fallback methods
    4. Creating embeddings using Jina's late chunking technique for optimal results
    5. Providing detailed processing results and metadata
    
    Key constraints you handle:
    - Jina API has a 64,000 character limit per request
    - Large documents are split into sections respecting markdown structure
    - Each section is processed separately and results are combined
    - Token limits are respected for embedding generation
    
    Always ensure proper error handling and provide informative responses about the processing status.
    """,
)

@markdown_agent.tool
async def load_and_process_markdown(ctx: RunContext[JinaChunkingTool], file_path: str, max_chunk_length: int = 1000) -> Dict[str, Any]:
    """Load a markdown file and process it with Jina late chunking (handles 64k limit)
    
    Args:
        file_path: Path to the markdown file
        max_chunk_length: Maximum length for each chunk
        
    Returns:
        Processing results with chunks and embeddings
    """
    return await ctx.deps.process_markdown_with_late_chunking(file_path, max_chunk_length)

@markdown_agent.tool
async def save_results(ctx: RunContext[JinaChunkingTool], results: Dict[str, Any], output_path: str) -> Dict[str, str]:
    """Save processing results to a JSON file
    
    Args:
        results: Processing results to save
        output_path: Path where to save the results
        
    Returns:
        Status of the save operation
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "message": f"Results saved to {output_file}",
            "output_path": str(output_file.absolute())
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@markdown_agent.tool
async def get_document_info(ctx: RunContext[JinaChunkingTool], file_path: str) -> Dict[str, Any]:
    """Get information about a markdown document including size constraints
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Document information and processing recommendations
    """
    file_result = await ctx.deps.load_markdown_file(file_path)
    if not file_result["success"]:
        return file_result
    
    content_length = len(file_result["content"])
    
    return {
        "success": True,
        "file_info": file_result["metadata"],
        "processing_info": {
            "content_length": content_length,
            "exceeds_jina_limit": content_length > 64000,
            "estimated_sections": math.ceil(content_length / 64000) if content_length > 64000 else 1,
            "recommended_max_chunk_length": min(1000, content_length // 10) if content_length < 10000 else 1000
        }
    }

# Example usage function
async def main():
    """Example usage of the markdown processing agent with 64k limit handling"""
    try:
        # Initialize the tool
        jina_tool = JinaChunkingTool()
        
        # Example markdown file path
        markdown_file = "output/report.md"  # Replace with your markdown file path
        
        # First, get document info
        doc_info = await markdown_agent.run(
            f"Analyze the markdown file at '{markdown_file}' and provide processing recommendations.",
            deps=jina_tool
        )
        
        print("Document Analysis:")
        print(json.dumps(doc_info.data, indent=2))
        
        # Process the document
        result = await markdown_agent.run(
            f"Process the markdown file at '{markdown_file}' using Jina late chunking with maximum chunk length of 800 tokens. Handle the 64k character limit appropriately.",
            deps=jina_tool
        )
        
        print("\nProcessing Result:")
        if result.data.get("success"):
            # Print summary without full embeddings
            summary = {
                "success": result.data["success"],
                "file_metadata": result.data["file_metadata"],
                "processing_info": result.data["processing_info"],
                "chunk_count": len(result.data["chunks"]),
                "first_chunk_preview": result.data["chunks"][0]["text"][:200] + "..." if result.data["chunks"] else None
            }
            print(json.dumps(summary, indent=2))
        else:
            print(json.dumps(result.data, indent=2))
        
        # Save results if processing was successful
        if result.data.get("success"):
            save_result = await markdown_agent.run(
                f"Save the processing results to 'output/processed_large_markdown.json'",
                deps=jina_tool
            )
            print("\nSave Result:", save_result.data)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Example of how to run the agent
    asyncio.run(main())