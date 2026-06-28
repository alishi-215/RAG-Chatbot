# RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot built with Flask, ChromaDB, and Azure OpenAI that answers questions from uploaded documents.

## Tech Stack
- **Backend:** Python, Flask
- **Vector DB:** ChromaDB
- **AI:** Azure OpenAI (GPT-4o-mini + text-embedding-3-small)
- **Document Support:** PDF, DOCX, TXT

## Features
- Upload and process documents (PDF, DOCX, TXT)
- Paragraph-based intelligent chunking
- Vector similarity search
- Context-aware Q&A from documents
- Persistent session memory
- Dynamic user info extraction

## Setup
### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment
Create a `.env` file:
AZURE_OPENAI_API_KEY=your_key

AZURE_OPENAI_ENDPOINT=your_endpoint

AZURE_API_VERSION=your_version

GPT_ENDPOINT=your_gpt_endpoint

GPT_API_KEY=your_gpt_key
### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment
Create a `.env` file:
