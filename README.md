# HR Policy RAG Application

A Retrieval-Augmented Generation (RAG) system that enables employees to query Company policies using natural language. The system processes HR policy documents and provides accurate, context-aware answers about leave policies, health insurance coverage, and other HR-related topics based on the provided documents.
## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │ -> │  Text Processing │ -> │ Vector Database │
│                 │    │  & Chunking      │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                              │
         v                                              v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │ -> │  Vector Search   │ -> │  LLM Response   │
│                 │    │  & Retrieval     │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Evaluation Architecture

```
┌─────────────────────┐
│    RAG Output       │
│    + User Query     │ 
│ + Retrieved context │ 
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│ Evaluation Pipeline │
│  (LangChain &/or    │
│   Custom Prompt)    │
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│   Scoring &         │
│   Reasoning Output  │
└─────────────────────┘
```

- RAG output (answers) are passed to the evaluation pipeline.
- Evaluation uses LangChain evaluators or custom LLM-based prompts.
- Results include automated scoring, reasoning, and pass/fail feedback.

## Project Structure

```
RAG_APP/
├── .env                                    # Environment variables
├── requirements.txt                        # Python dependencies
├── README.md                              # Project documentation
│
├── data/
│   ├── raw_pdfs/                          # Original PDF documents
│   ├── texts/                             # Extracted text files
│   └── chroma_db/                         # Vector database storage
│       └── chroma.sqlite3
│
├── src/
│   ├── Indexing/                          # Document processing pipeline
│   │   ├── loading_documents.py           # PDF text extraction
│   │   ├── text_splitting.py              # Document chunking
│   │   └── embeddings.py                  # Vector embeddings creation
│   ├── retrieval_generation/              # Query processing
│   │   └── query_rag.py                   # RAG chain implementation
│   └── evaluation/                        # System evaluation tools
│       ├── langchain_evaluator.py         # LangChain-based evaluation
│       ├── prompt_evaluator.py            # Custom prompt evaluation
│       ├── run_eval.py                    # Langchain evaluation runner
│       └── run_prompt_eval.py             # Prompt evaluation runner
│
└── env/                                   # Virtual environment
```

## Getting Started

### Prerequisites

- Python 3.8+
- Fireworks API key for LLM access
- Unstructured API key for OCR

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/minahilsana/RAG_APP.git
   cd RAG_APP
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   FIREWORKS_API_KEY=your_fireworks_api_key_here
   UNSTRUCTURED_API_KEY=your_unstructured_api_key_here
   ```

### Initial Setup

1. **Process documents and create vector database**
   ```python
   from src.Indexing.text_splitting import split
   from src.Indexing.embeddings import create_and_store_embeddings
   
   # Split documents into chunks
   chunks = split()
   
   # Create and store embeddings
   create_and_store_embeddings(chunks)
   ```

## Usage

### Interactive CLI

Run the interactive query interface:

```bash
python -m src.retrieval_generation.query_rag
```

## Core Components

### 1. Document Processing Pipeline

#### **Text Extraction** ([`src/Indexing/loading_documents.py`](src/Indexing/loading_documents.py))
- Extracts text from PDF documents
- Handles OCR for scanned documents (optional)
- Saves processed text to [`data/texts/`](data/texts/)

#### **Text Chunking** ([`src/Indexing/text_splitting.py`](src/Indexing/text_splitting.py))
- Uses [`RecursiveCharacterTextSplitter`](src/Indexing/text_splitting.py)
- **Chunk size**: 1000 characters
- **Overlap**: 200 characters for context preservation
- Creates LangChain Document objects with metadata

#### **Vector Embeddings** ([`src/Indexing/embeddings.py`](src/Indexing/embeddings.py))
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Vector Database**: ChromaDB
- **Storage**: [`data/chroma_db/`](data/chroma_db/)

### 2. Query Processing

#### **RAG Chain** ([`src/retrieval_generation/query_rag.py`](src/retrieval_generation/query_rag.py))
- **Retriever**: ChromaDB vector similarity search
- **LLM**: Fireworks `llama-v3p1-8b-instruct`
- **Context Processing**: Raw and formatted versions
- **Response Generation**: Grounded in retrieved documents

## Evaluation System

The application includes comprehensive evaluation tools to assess RAG performance:

### 1. LangChain Evaluators ([`src/evaluation/langchain_evaluator.py`](src/evaluation/langchain_evaluator.py))

Uses built-in LangChain evaluation criteria:

- **Conciseness**: Response brevity and clarity
- **Correctness**: Factual accuracy against context
- **Relevance**: Alignment with user question


**Usage:**
```bash
python -m src.evaluation.run_eval
```

### 2. Custom Prompt Evaluation ([`src/evaluation/prompt_evaluator.py`](src/evaluation/prompt_evaluator.py))

Comprehensive evaluation using a specialized LLM evaluator:

#### Evaluation Criteria:
- **Response Validation**: Complete question answering
- **Response Consistency**: Logical coherence
- **Correctness**: Factual alignment with context
- **Contextual Relevance**: Context grounding
- **Hallucination Detection**: Information accuracy
- **Overall Score**: Pass/Fail assessment

#### Evaluator Configuration:
- **Model**: `llama-v3p1-70b-instruct` 
- **Output**: Structured JSON with detailed reasoning
- **Temperature**: 0.0 for consistent evaluation

**Usage:**
```bash
python -m src.evaluation.run_prompt_eval
```

### 3. Evaluation Metrics

Both evaluation systems provide:

- **Automated scoring** with Pass/Fail normalization
- **Detailed reasoning** for each criterion
- **Interactive testing** with real-time feedback
- **Formatted output** with color-coded results

## Technical Configuration

### Model Settings

**Embedding Model:**
- `sentence-transformers/all-mpnet-base-v2`
- Local processing (no API required)

**Generation Model:**
- `accounts/fireworks/models/llama-v3p1-8b-instruct`
- Temperature: 0.1 (focused responses)
- Max tokens: 1024

**Evaluation Model:**
- `accounts/fireworks/models/llama-v3p1-70b-instruct`
- Temperature: 0.0 (consistent evaluation)
- Max tokens: 512

### Database Configuration

**ChromaDB Settings:**
- **Persistence**: Enabled at [`data/chroma_db/`](data/chroma_db/)
- **Collection**: Policy documents with metadata
- **Search**: Similarity-based retrieval
- **Metadata**: Source document tracking

## Customization

### Adding New Documents

1. Place PDF files in [`data/raw_pdfs/`](data/raw_pdfs/)
2. Run text extraction and chunking
3. Recreate embeddings database
4. Test with relevant queries

### Modifying Chunk Settings

Edit [`src/Indexing/text_splitting.py`](src/Indexing/text_splitting.py):
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # Increase for longer context
    chunk_overlap=300,      # Adjust overlap as needed
    length_function=len,
    is_separator_regex=False
)
```

### Evaluation Customization

Modify evaluation criteria in [`src/evaluation/prompt_evaluator.py`](src/evaluation/prompt_evaluator.py):
```python
# Add custom evaluation criteria
evaluation_prompt = PromptTemplate.from_template("""
    Custom evaluation criteria:
    - Policy Compliance: Does the response align with company policies?
    - Actionability: Does the response provide clear next steps?
    ...
""")
```
### Performance Optimization

- **Chunk Size**: Adjust based on document complexity
- **Retrieval Count**: Modify `k` parameter in similarity search
- **Model Selection**: Use different LLM models based on requirements
