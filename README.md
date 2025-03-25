# Document Processing Service

A powerful document processing service that enables natural language querying of PDF documents using RAG (Retrieval-Augmented Generation) technology. Built with Docling, LlamaIndex, and Sentence Transformers.

## Features

- **PDF Processing**

  - Text extraction from PDF documents using Docling
  - Form field extraction and validation
  - OCR support for scanned documents
  - Metadata extraction

- **Document Processing**

  - Intelligent text chunking with LlamaIndex
  - Text cleaning and normalization
  - Form field validation
  - Image-based document detection

- **Vector Storage**

  - Pinecone integration for vector storage
  - Efficient document indexing
  - Semantic search capabilities
  - Configurable chunking and overlap

- **Query Processing**

  - Natural language querying with RAG
  - Semantic search using Sentence Transformers
  - Context-aware responses
  - Query validation and cleaning

- **Performance & Monitoring**
  - Performance monitoring
  - Detailed logging
  - Error tracking
  - Health checks

## Architecture

```
services/app/
├── api/                    # API endpoints and routes
├── core/                   # Core application logic
├── models/                 # Data models
│   └── document.py        # Document-related models
├── services/              # Service modules
│   ├── document_processor.py  # PDF processing with Docling
│   ├── form_extractor.py     # Form field extraction
│   ├── text_processor.py     # Text processing
│   ├── document_pipeline.py  # Main RAG pipeline
├── exceptions/            # Custom exceptions
│   └── validation.py     # Validation exceptions
├── config.py             # Application configuration
├── logger_config.py      # Logging configuration
├── validators.py         # Validation functions
└── main.py              # FastAPI application
```

## Prerequisites

1. **Python 3.8 or higher**
2. **Pinecone Account**
   - Sign up at https://www.pinecone.io/
   - Get your API key and environment

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd docling-rag-app
   ```

2. **Create and activate virtual environment**

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   # Install application dependencies
   pip install -r services/app/requirements.txt

   # Install test dependencies (optional)
   pip install -r services/app/tests/requirements-test.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the `services/app` directory:

   ```env
   # Pinecone Configuration
   PINECONE_API_KEY=your_api_key
   PINECONE_ENVIRONMENT=your_environment
   PINECONE_INDEX_NAME=your_index_name

   # File Configuration
   UPLOAD_DIR=uploads
   MAX_FILE_SIZE=10485760  # 10MB in bytes

   # Logging Configuration
   LOG_LEVEL=INFO
   LOG_FILE=app.log
   MAX_LOG_SIZE=10485760  # 10MB in bytes
   LOG_BACKUP_COUNT=5

   # Document Processing
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200

   # Validation
   MIN_QUERY_LENGTH=2
   ALLOWED_FILE_TYPES=pdf

   # Performance Monitoring
   ENABLE_PERFORMANCE_MONITORING=true

   # Cache Configuration
   ENABLE_CACHING=true
   CACHE_TTL=3600  # 1 hour in seconds
   ```

## Running the Application

### Local Development

1. **Start the server**

   ```bash
   # Navigate to the app directory
   cd services/app

   # Start the server with uvicorn
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the API**

   - API Documentation: http://localhost:8000/docs
   - Alternative Documentation: http://localhost:8000/redoc
   - Health Check: http://localhost:8000/health

3. **Test the API**

   ```bash
   # Upload a document
   curl -X POST "http://localhost:8000/upload/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@/path/to/your/document.pdf"

   # Query a document
   curl -X POST "http://localhost:8000/query/" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{"filename": "document.pdf", "query": "What is the main topic?"}'
   ```

### Docker Deployment

1. **Build the Docker image**

   ```bash
   # From the project root
   docker build -t docling-rag-app -f services/app/Dockerfile .
   ```

2. **Run the container**

   ```bash
   docker run -d \
     --name docling-rag-app \
     -p 8000:8000 \
     -v $(pwd)/services/app/uploads:/app/uploads \
     -v $(pwd)/services/app/logs:/app/logs \
     --env-file services/app/.env \
     docling-rag-app
   ```

3. **Access the application**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

### Kubernetes Deployment

1. **Create namespace**

   ```bash
   kubectl create namespace docling-rag
   ```

2. **Create secrets**

   ```bash
   # Create secret from .env file
   kubectl create secret generic docling-rag-secrets \
     --from-file=.env=services/app/.env \
     -n docling-rag
   ```

3. **Deploy the application**

   ```bash
   # Apply Kubernetes manifests
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   ```

4. **Verify deployment**

   ```bash
   # Check pods
   kubectl get pods -n docling-rag

   # Check services
   kubectl get svc -n docling-rag

   # Check ingress
   kubectl get ingress -n docling-rag
   ```

5. **Access the application**
   - Get the external IP/hostname
   ```bash
   kubectl get ingress -n docling-rag -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}'
   ```
   - Access via: http://<external-ip>

### Development Workflow

1. **Set up development environment**

   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate

   # Install development dependencies
   pip install -r services/app/tests/requirements-test.txt
   ```

2. **Run tests**

   ```bash
   # Run all tests
   pytest services/app/tests/

   # Run with coverage
   pytest --cov=app services/app/tests/
   ```

3. **Code quality checks**

   ```bash
   # Format code
   black .

   # Check types
   mypy .

   # Lint code
   flake8
   ```

4. **Local development with hot reload**

   ```bash
   # Start server with hot reload
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

   # In another terminal, run tests in watch mode
   pytest-watch services/app/tests/
   ```

### Monitoring and Debugging

1. **View logs**

   ```bash
   # Local development
   tail -f services/app/logs/app.log

   # Docker
   docker logs -f docling-rag-app

   # Kubernetes
   kubectl logs -f deployment/docling-rag-app -n docling-rag
   ```

2. **Access metrics**

   ```bash
   # Local development
   curl http://localhost:8000/metrics

   # Kubernetes
   kubectl port-forward svc/docling-rag-app 8000:8000 -n docling-rag
   ```

3. **Debug with VS Code**
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: FastAPI",
         "type": "python",
         "request": "launch",
         "module": "uvicorn",
         "args": [
           "main:app",
           "--reload",
           "--host",
           "0.0.0.0",
           "--port",
           "8000"
         ],
         "jinja": true,
         "justMyCode": true
       }
     ]
   }
   ```

## RAG System Components

### 1. Document Processing (Docling)

- PDF text extraction
- OCR for scanned documents
- Metadata extraction
- Form field processing

### 2. Text Processing (LlamaIndex)

- Intelligent text chunking
- Node parsing
- Document structuring
- Context preservation

### 3. Embeddings (Sentence Transformers)

- Using `all-MiniLM-L6-v2` model
- 384-dimensional embeddings
- Local processing (no API calls)
- Efficient semantic understanding

### 4. Vector Storage (Pinecone)

- Efficient vector indexing
- Cosine similarity search
- Scalable storage
- Fast retrieval

### 5. Query Processing

- Semantic search
- Context-aware responses
- Tree-based summarization
- Top-k similarity retrieval

## Development

### Running Tests

```bash
# Run all tests
pytest services/app/tests/

# Run with coverage
pytest --cov=app services/app/tests/

# Run specific test file
pytest services/app/tests/test_document_processor.py

# Run with verbose output
pytest -v services/app/tests/
```

### Code Style

```bash
# Format code
black .

# Check types
mypy .

# Lint code
flake8
```

## API Endpoints

### Upload Document

```http
POST /upload/
Content-Type: multipart/form-data

file: <PDF file>
```

Response:

```json
{
  "message": "Document processed successfully",
  "filename": "example.pdf",
  "metadata": {
    "title": "Example Document",
    "author": "John Doe",
    "subject": "Test Document",
    "creation_date": "2024-03-20T10:00:00",
    "modification_date": "2024-03-20T10:00:00",
    "page_number": 1,
    "total_pages": 1,
    "form_data": {},
    "processing_timestamp": "2024-03-20T10:00:00"
  }
}
```

### Query Document

```http
POST /query/
Content-Type: application/json

{
    "filename": "example.pdf",
    "query": "What is the main topic of this document?"
}
```

Response:

```json
{
  "query": "What is the main topic of this document?",
  "response": "The main topic of this document is...",
  "processing_time": "2024-03-20T10:00:00"
}
```

### Health Check

```http
GET /health/
```

Response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Document Processing Pipeline

1. **Document Loading**

   - PDF validation
   - File size check
   - Format verification

2. **Content Extraction**

   - Text extraction with Docling
   - Form field extraction
   - OCR for scanned documents
   - Metadata extraction

3. **Text Processing**

   - Text cleaning
   - Chunking with LlamaIndex
   - Node parsing
   - Metadata enrichment

4. **Vector Storage**

   - Document indexing
   - Embedding generation with Sentence Transformers
   - Pinecone integration
   - Cache management

5. **Query Processing**
   - Query validation
   - Semantic search
   - Context retrieval
   - Response generation

## Logging

The application uses a rotating file handler for logging:

- Log file: `app.log`
- Maximum size: 10MB
- Backup count: 5 files
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Error Handling

The application implements comprehensive error handling:

- Custom validation exceptions
- HTTP error responses
- Detailed error logging
- Graceful failure handling

## Performance Monitoring

Performance metrics are tracked for:

- Document processing time
- Query response time
- OCR processing time
- Vector store operations
- Cache hit/miss rates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]
