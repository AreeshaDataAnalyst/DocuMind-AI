# DocuMind-AI
RAG-based PDF Question Answering Chatbot

# DocuMind AI - PDF Study Assistant

## Project Overview

DocuMind AI is a Retrieval-Augmented Generation
(RAG) application that allows users to upload
PDF documents and ask questions in natural language.

The application retrieves relevant document content
using vector similarity search and generates answers
using Google Gemini.

## Features

- PDF-only upload
- Text-based PDF validation
- PDF text extraction
- Text chunking with overlap
- Sentence embeddings
- FAISS vector search
- Gemini-powered question answering
- Source page references
- Streamlit user interface

## Technologies

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- FAISS
- PyMuPDF
- NumPy

## Project Workflow

1. Upload a PDF document.
2. Validate the PDF.
3. Extract text.
4. Divide text into chunks.
5. Generate embeddings.
6. Store vectors in FAISS.
7. Retrieve relevant chunks.
8. Generate a document-based answer.

## Installation

Clone the repository:

```bash
[git clone YOUR_GITHUB_REPOSITORY_URL](https://github.com/AreeshaDataAnalyst/DocuMind-AI)
```

Open the project folder:

```bash
cd DocuMind-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Configuration

Create:

`.streamlit/secrets.toml`

Add your Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Never upload your API key to GitHub.

## Run Application

```bash
python -m streamlit run app.py
```

## Limitations

The application supports text-based PDFs.
Scanned and image-only PDFs are rejected using
text and image validation checks.

Some unusual image-based PDFs may require
additional validation methods.

## Author

Areesha Atique

Computational Mathematics Graduate
