# 🚀 Custom RAG System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/yourusername/custom-rag-system)

A powerful and customizable Retrieval-Augmented Generation (RAG) system built with Python, featuring advanced text chunking, semantic embeddings, and vector search capabilities powered by Pinecone.

## ✨ Features

- 🔍 **Advanced Text Chunking**: Intelligent text segmentation with configurable chunk sizes and overlap
- 🧠 **Semantic Embeddings**: State-of-the-art sentence transformers for high-quality vector representations
- 🔎 **Vector Search**: Lightning-fast similarity search using Pinecone's vector database
- 📊 **Data Management**: Efficient data import, processing, and upserting capabilities
- 🚀 **Scalable Architecture**: Modular design for easy customization and extension
- 📈 **Performance Optimized**: Built for production use with progress tracking and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Text Chunker   │───▶│ Embedding Model │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Search  │◀───│  Pinecone DB    │◀───│  Data Upserter  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Pinecone account and API key
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/custom-rag-system.git
   cd custom-rag-system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Pinecone API key and configuration
   ```

### Configuration

Create a `.env` file with your configuration:

```env
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=your_index_name
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📖 Usage

### 1. Create Pinecone Index

```python
from create_index import create_pinecone_index

# Create a new index for your embeddings
create_pinecone_index(
    index_name="my-rag-index",
    dimension=384,  # Dimension based on your embedding model
    metric="cosine"
)
```

### 2. Import and Process Data

```python
from data_importer import DataImporter
from text_chunker import TextChunker
from embedding_model import EmbeddingModel

# Initialize components
importer = DataImporter()
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
embedding_model = EmbeddingModel()

# Import and process documents
documents = importer.load_documents("path/to/your/documents")
chunks = chunker.chunk_documents(documents)
embeddings = embedding_model.generate_embeddings(chunks)
```

### 3. Store in Pinecone

```python
from data_upserter import DataUpserter

# Initialize Pinecone client and upsert data
upserter = DataUpserter()
upserter.upsert_embeddings(embeddings, metadata=chunks)
```

### 4. Perform Vector Search

```python
from vector_search import VectorSearch

# Initialize search and perform queries
searcher = VectorSearch()
results = searcher.search(
    query="What is machine learning?",
    top_k=5,
    include_metadata=True
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.metadata['text']}")
    print("---")
```

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
python test_rag_system.py
```

## 📁 Project Structure

```
custom-rag-system/
├── 📄 create_index.py          # Pinecone index creation utilities
├── 📄 data_importer.py         # Data loading and preprocessing
├── 📄 data_upserter.py         # Data storage in Pinecone
├── 📄 embedding_model.py       # Text embedding generation
├── 📄 pinecone_client.py       # Pinecone client configuration
├── 📄 text_chunker.py          # Text segmentation logic
├── 📄 vector_search.py         # Vector similarity search
├── 📄 test_rag_system.py       # Comprehensive test suite
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env.example             # Environment variables template
├── 📄 .gitignore               # Git ignore rules
└── 📄 README.md                # This file
```

## 🔧 Customization

### Text Chunking

Modify chunking parameters in `text_chunker.py`:

```python
chunker = TextChunker(
    chunk_size=1500,      # Adjust chunk size
    chunk_overlap=300,    # Adjust overlap
    separator="\n\n"      # Custom separator
)
```

### Embedding Models

Change the embedding model in `embedding_model.py`:

```python
# Available models:
# - all-MiniLM-L6-v2 (384 dimensions, fast)
# - all-mpnet-base-v2 (768 dimensions, accurate)
# - multi-qa-MiniLM-L6-cos-v1 (384 dimensions, QA optimized)
```

### Search Parameters

Customize search behavior in `vector_search.py`:

```python
results = searcher.search(
    query="your query",
    top_k=10,                    # Number of results
    include_metadata=True,        # Include metadata
    filter={"category": "tech"}   # Add filters
)
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Pinecone](https://www.pinecone.io/) for vector database infrastructure
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Hugging Face](https://huggingface.co/) for pre-trained models

## 📞 Support

If you have any questions or need help, please:

- 📖 Check the [documentation](docs/)
- 🐛 Open an [issue](https://github.com/yourusername/custom-rag-system/issues)
- 💬 Start a [discussion](https://github.com/yourusername/custom-rag-system/discussions)

---

<div align="center">
  <p>Made with ❤️ for the AI community</p>
  <p>⭐ Star this repo if you found it helpful!</p>
  <p>Uncle Engineer | Custom AI with RAG course</p>
</div>
