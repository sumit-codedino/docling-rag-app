from docling.document_converter import DocumentConverter
from pathlib import Path
import nest_asyncio
from dotenv import load_dotenv
import os
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer, BatchEncoding
import requests
import torch
import numpy as np
import math
import json

# Configure logging
def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger with console handler only
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

# Load environment variables
load_dotenv()
nest_asyncio.apply()

def convert_to_markdown(pdf_path: str) -> Dict[str, Any]:
    """Convert PDF to markdown using docling
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dict containing markdown content and metadata
    """
    try:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            logger.error(f"PDF file not found: {pdf_path_obj}")
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path_obj}",
                "source_file": str(pdf_path_obj)
            }

        if not pdf_path_obj.suffix.lower() == '.pdf':
            logger.error(f"Invalid file type: {pdf_path_obj.suffix}")
            return {
                "success": False,
                "error": f"Invalid file type: {pdf_path_obj.suffix}. Only PDF files are supported.",
                "source_file": str(pdf_path_obj)
            }

        logger.info(f"Starting PDF conversion: {pdf_path_obj}")
        start_time = datetime.now()

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path_obj))
        markdown = result.document.export_to_markdown()

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"PDF conversion completed: {pdf_path_obj} in {processing_time:.2f} seconds")

        return {
            "success": True,
            "content": markdown,
            "metadata": {
                "source_file": str(pdf_path_obj),
                "file_size": pdf_path_obj.stat().st_size,
                "processing_time": processing_time,
                "content_length": len(markdown),
                "converted_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error converting PDF to markdown: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "source_file": pdf_path
        }

def store_markdown_file(markdown_text: str, file_name: str) -> Dict[str, Any]:
    """Store markdown content in a file
    
    Args:
        markdown_text: Markdown content to store
        file_name: Name of the file to store the markdown content in
        
    Returns:
        Dict containing storage result and metadata
    """
    try:
        if not markdown_text:
            logger.error("Empty markdown content provided")
            return {
                "success": False,
                "error": "Empty markdown content provided"
            }

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Ensure file_name has .md extension
        if not file_name.endswith('.md'):
            file_name = f"{file_name}.md"
            
        file_path = output_dir / file_name
        
        # Check if file already exists
        if file_path.exists():
            logger.warning(f"File already exists: {file_path}")
            backup_path = output_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            file_path.rename(backup_path)
            logger.info(f"Created backup of existing file: {backup_path}")
        
        # Write the markdown content
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(markdown_text)
            
        file_size = file_path.stat().st_size
        logger.info(f"Markdown file stored: {file_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "file_path": str(file_path),
            "file_size": file_size,
            "stored_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error storing markdown file: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "file_name": file_name
        }
    
def custom_tokenize_jina_api(input_text: str):
    url = 'https://segment.jina.ai/'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
    }
    data = {
        "content": input_text,
        "tokenizer": "o200k_base",
        "return_tokens": "true",
        "return_chunks": "true",
        "max_chunk_length": "1000"
    }
    # Make the API request
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    chunks = response_data.get("chunks", [])
    i = 1
    j = 1
    span_annotations = []
    for x in response_data['tokens']:
        if j == 1:
            j = len(x)
        else:
            j = len(x) + i
        span_annotations.append((i, j))
        i = j
    return chunks, span_annotations


        
async def process_pdf(pdf_path: str, output_name: str) -> Dict[str, Any]:
    """Complete pipeline: convert PDF to markdown and store it
    
    Args:
        pdf_path: Path to the PDF file
        output_name: Name for the output markdown file
        
    Returns:
        Dict containing processing results
    """
    try:
        # Step 1: Convert PDF to markdown
        conversion_result = convert_to_markdown(pdf_path)
        if not conversion_result["success"]:
            return conversion_result
            
        # Step 2: Store markdown file
        storage_result = store_markdown_file(conversion_result["content"], output_name)
        if not storage_result["success"]:
            return storage_result
            
        return {
            "success": True,
            "conversion": conversion_result["metadata"],
            "storage": storage_result,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in PDF processing pipeline: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "source_file": pdf_path
        }
    
if __name__ == "__main__":
    try:
        logger.info("Starting PDF ingestion process")
        
        # Get file path from environment or use default
        pdf_path = os.getenv("PDF_FILE", "emami.pdf")
        output_name = os.getenv("OUTPUT_NAME", "emami")
        
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Output name: {output_name}")
        
        # Process the PDF
        result = asyncio.run(process_pdf(pdf_path, output_name))
        
        if result["success"]:
            logger.info("PDF processing completed successfully")
            print("\nProcessing Summary:")
            print(f"Source: {result['conversion']['source_file']}")
            print(f"Output: {result['storage']['file_path']}")
            print(f"Processing time: {result['conversion']['processing_time']:.2f} seconds")
            print(f"Content length: {result['conversion']['content_length']} characters")
            print(f"File size: {result['storage']['file_size']} bytes")
        else:
            logger.error(f"PDF processing failed: {result['error']}")
            print(f"\nError: {result['error']}")
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"Fatal error: {str(e)}")
        exit(1)