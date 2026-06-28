import os
import json
import secrets
import uuid
import time
import re
from datetime import datetime
from typing import Dict, List, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chromadb import PersistentClient
import pypdf
from docx import Document
import requests
import logging
import sys

# ==================== CONFIGURATION ====================
class Config:
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = ""
    AZURE_API_KEY = ""
    API_VERSION = ""
    
    GPT_ENDPOINT = ""
    GPT_API_KEY = ""
    GPT_API_VERSION = ""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
    
    # Embedding
    EMBEDDING_DIMENSION = 1536
    
    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate_config(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.CHROMA_PATH, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        if not os.path.exists(cls.FRONTEND_DIR):
            print(f"⚠️ Warning: Frontend directory not found at {cls.FRONTEND_DIR}")
            cls.FRONTEND_DIR = os.path.join(cls.BASE_DIR)

config = Config()
config.validate_config()

# ==================== SIMPLE LOGGING ====================
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(
        os.path.join(config.LOGS_DIR, f"rag_only_{datetime.now().strftime('%Y%m%d')}.log"),
        encoding='utf-8'
    )
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def log_info(message: str, **kwargs):
    extra = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    if extra:
        logger.info(f"{message} | {extra}")
    else:
        logger.info(message)

def log_error(message: str, **kwargs):
    extra = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    if extra:
        logger.error(f"{message} | {extra}")
    else:
        logger.error(message)

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(32)

# ==================== CHROMADB INITIALIZATION ====================
def initialize_chromadb():
    try:
        log_info("Initializing ChromaDB")
        
        client = PersistentClient(path=config.CHROMA_PATH)
        
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        log_info(f"Found collections: {collection_names}")
        
        try:
            collection = client.get_collection(name="documents")
            count = collection.count()
            log_info(f"Using existing collection 'documents' with {count} items")
        except Exception:
            log_info("Creating new collection 'documents'")
            collection = client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine", "dimension": config.EMBEDDING_DIMENSION}
            )
        
        return client, collection
        
    except Exception as e:
        log_error(f"Error initializing ChromaDB", error=str(e))
        return None, None

client, collection = initialize_chromadb()

# ==================== DOCUMENT PROCESSING ====================
class DocumentProcessor:
    @staticmethod
    def clean_document_text(text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'\{#[^}]+\}', '', text)
        text = re.sub(r'\.[A-Za-z\-]+\}', '', text)
        text = re.sub(r'\{\.TOC-Heading\}', '', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line in ['#', '##', '###']:
                continue
            if line.startswith('# ') and len(line) < 10:
                continue
            if 'TOC-Heading' in line:
                continue
            if line and len(line) > 3:
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_num + 1} of {total_pages}:\n{page_text}\n\n"
                
                text = DocumentProcessor.clean_document_text(text)
                log_info("PDF processed", file_path=file_path, pages=total_pages)
                
        except Exception as e:
            log_error("Error reading PDF", error=str(e), file_path=file_path)
        
        return text
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            text = DocumentProcessor.clean_document_text(text)
            log_info("DOCX processed", file_path=file_path)
            
        except Exception as e:
            log_error("Error reading DOCX", error=str(e), file_path=file_path)
        
        return text
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text = DocumentProcessor.clean_document_text(text)
            log_info("TXT processed", file_path=file_path)
            
            return text
        except Exception as e:
            log_error("Error reading TXT", error=str(e), file_path=file_path)
            return ""
    
    @staticmethod
    def chunk_by_paragraphs(text: str, 
                           max_chunk_size: int = config.CHUNK_SIZE,
                           overlap: int = config.CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        if not text or len(text.strip()) == 0:
            return []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        log_info("Starting paragraph-based chunking", total_paragraphs=len(paragraphs))
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 1
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_length = len(paragraph)
            
            if para_length > max_chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': {
                            'chunk_id': chunk_id,
                            'chunk_type': 'paragraph_group'
                        }
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    sent_len = len(sentence)
                    if sentence_length + sent_len > max_chunk_size and sentence_chunk:
                        chunks.append({
                            'text': ' '.join(sentence_chunk),
                            'metadata': {
                                'chunk_id': chunk_id,
                                'chunk_type': 'sentence_group'
                            }
                        })
                        chunk_id += 1
                        sentence_chunk = [sentence]
                        sentence_length = sent_len
                    else:
                        sentence_chunk.append(sentence)
                        sentence_length += sent_len
                
                if sentence_chunk:
                    chunks.append({
                        'text': ' '.join(sentence_chunk),
                        'metadata': {
                            'chunk_id': chunk_id,
                            'chunk_type': 'sentence_group'
                        }
                    })
                    chunk_id += 1
            
            else:
                if current_length + para_length > max_chunk_size and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': {
                            'chunk_id': chunk_id,
                            'chunk_type': 'paragraph_group'
                        }
                    })
                    chunk_id += 1
                    
                    overlap_paras = current_chunk[-overlap:] if overlap > 0 else []
                    current_chunk = overlap_paras + paragraph.split()
                    current_length = len(' '.join(current_chunk))
                else:
                    current_chunk.extend(paragraph.split())
                    current_length += para_length + 1
        
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': {
                    'chunk_id': chunk_id,
                    'chunk_type': 'paragraph_group'
                }
            })
        
        log_info("Paragraph chunking completed", total_chunks=len(chunks))
        
        return chunks
    
    @staticmethod
    def process_file(file_path: str, filename: str) -> tuple:
        text = ""
        
        if filename.lower().endswith('.pdf'):
            text = DocumentProcessor.read_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            text = DocumentProcessor.read_docx(file_path)
        elif filename.lower().endswith('.txt'):
            text = DocumentProcessor.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        if not text or len(text.strip()) == 0:
            raise ValueError("Could not extract text from file")
        
        chunks = DocumentProcessor.chunk_by_paragraphs(text)
        
        return chunks, text

# ==================== AZURE OPENAI FUNCTIONS ====================
def get_azure_embedding(text: str, max_retries: int = 3):
    if not text or len(text.strip()) < 10:
        return None
    
    for attempt in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": config.AZURE_API_KEY
            }
            
            text_for_embedding = text[:8000]
            data = {"input": text_for_embedding}
            
            response = requests.post(
                f"{config.AZURE_OPENAI_ENDPOINT}?api-version={config.API_VERSION}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                    if len(embedding) == config.EMBEDDING_DIMENSION:
                        log_info("Got embedding", text_length=len(text))
                        return embedding
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        except Exception as e:
            log_error(f"Embedding attempt failed", attempt=attempt+1, error=str(e))
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    log_error(f"Failed to get embedding after {max_retries} attempts")
    return None

def call_gpt(messages: List[Dict], max_tokens: int = 500):
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": config.GPT_API_KEY
        }
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        log_info("Calling GPT", message_count=len(messages))
        
        response = requests.post(
            f"{config.GPT_ENDPOINT}?api-version={config.GPT_API_VERSION}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                answer = result['choices'][0]['message']['content'].strip()
                log_info("GPT response received", response_length=len(answer))
                return answer
            else:
                log_error("No choices in GPT response")
                return None
        else:
            log_error("GPT API error", status_code=response.status_code)
            return None
            
    except Exception as e:
        log_error("GPT call failed", error=str(e))
        return None

# ==================== FLASK ROUTES ====================
@app.route('/upload', methods=['POST'])
def upload_file():
    log_info("File upload requested")
    
    if 'file' not in request.files:
        log_error("No file in request")
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        log_error("Empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    log_info(f"Processing file upload: {file.filename}")
    
    file_path = os.path.join(config.UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        chunks, full_text = DocumentProcessor.process_file(file_path, file.filename)
        
        if not chunks:
            raise ValueError("No content chunks created from file")
        
        log_info(f"Document processed", filename=file.filename, total_chunks=len(chunks))
        
        successful_chunks = 0
        failed_chunks = 0
        
        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data['text']
            chunk_metadata = chunk_data['metadata']
            
            if not chunk_text.strip():
                continue
                
            embedding = get_azure_embedding(chunk_text)
            
            if embedding:
                chunk_id = f"{file.filename}_{i}_{uuid.uuid4().hex[:8]}"
                
                try:
                    full_metadata = {
                        "filename": file.filename,
                        "upload_time": datetime.now().isoformat(),
                        **chunk_metadata
                    }
                    
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk_text],
                        ids=[chunk_id],
                        metadatas=[full_metadata]
                    )
                    successful_chunks += 1
                    log_info(f"Chunk stored", chunk_id=chunk_id[:16])
                except Exception as e:
                    failed_chunks += 1
                    log_error(f"Failed to store chunk", chunk_index=i, error=str(e))
            else:
                failed_chunks += 1
        
        os.remove(file_path)
        
        if successful_chunks > 0:
            log_info(f"File upload successful", filename=file.filename, successful_chunks=successful_chunks)
            
            return jsonify({
                "success": True,
                "message": f"✅ Document processed successfully",
                "chunks_processed": successful_chunks,
                "chunks_failed": failed_chunks,
                "filename": file.filename,
                "total_chunks": len(chunks)
            })
        else:
            log_error(f"No chunks processed successfully", filename=file.filename)
            return jsonify({
                "error": f"Failed to process any chunks from the document."
            }), 500
            
    except Exception as e:
        log_error(f"Error processing file", error=str(e), filename=file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['GET'])
def chat():
    query = request.args.get('query', '').strip()
    if not query:
        log_error("Empty chat query")
        return jsonify({"error": "Please enter a question"}), 400
    
    log_info(f"Chat query received", query=query[:100])
    
    document_context = ""
    relevant_chunks = []
    query_embedding = get_azure_embedding(query)
    
    if query_embedding:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            if results and 'documents' in results and results['documents']:
                relevant_chunks = results['documents'][0] if results['documents'][0] else []
                
                if relevant_chunks:
                    log_info(f"Found relevant document chunks", chunk_count=len(relevant_chunks))
                    
                    context_parts = []
                    for i, chunk in enumerate(relevant_chunks):
                        context_parts.append(f"[Document chunk {i+1}]:\n{chunk}")
                    
                    document_context = "\n\n---\n\n".join(context_parts)
        except Exception as e:
            log_error(f"Error searching ChromaDB", error=str(e))
    
    try:
        messages = []
        
        if document_context:
            system_prompt = f"""You are a helpful assistant that answers questions based on provided documents.

DOCUMENT CONTEXT:
{document_context}

INSTRUCTIONS:
1. Answer based ONLY on the document context above
2. If the answer is not in the document, say so politely
3. Do not use any external knowledge
4. Be concise and accurate"""
        else:
            system_prompt = """You are a helpful assistant. I couldn't find any relevant information in the documents for your question. 
            Please ask about something related to the uploaded documents or upload a document first."""
        
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        answer = call_gpt(messages, max_tokens=500)
        
        if answer:
            response_data = {
                "answer": answer,
                "relevant_chunks": len(relevant_chunks),
                "found_in_document": bool(document_context),
                "from_memory": False
            }
            
            log_info(f"Response generated", answer_length=len(answer))
            
            return jsonify(response_data)
        else:
            return jsonify({"error": "Failed to generate response"}), 500
        
    except Exception as e:
        log_error(f"Error processing chat query", error=str(e))
        return jsonify({
            "error": "An error occurred while processing your query"
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    try:
        results = collection.get()
        if results and results['metadatas']:
            documents = {}
            for metadata in results['metadatas']:
                filename = metadata.get('filename', 'Unknown')
                if filename not in documents:
                    documents[filename] = 0
                documents[filename] += 1
            
            log_info(f"Documents listed", document_count=len(documents))
            
            return jsonify({
                "success": True,
                "documents": [
                    {"filename": name, "chunks": count}
                    for name, count in documents.items()
                ],
                "total_chunks": len(results['ids'])
            })
        else:
            return jsonify({
                "success": True,
                "documents": [],
                "message": "No documents uploaded yet",
                "total_chunks": 0
            })
    except Exception as e:
        log_error(f"Error listing documents", error=str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    try:
        results = collection.get()
        total_chunks = len(results['ids']) if results and results['ids'] else 0
        
        try:
            test_response = requests.post(
                f"{config.GPT_ENDPOINT}?api-version={config.GPT_API_VERSION}",
                headers={"api-key": config.GPT_API_KEY, "Content-Type": "application/json"},
                json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
                timeout=5
            )
            if test_response.status_code == 200:
                gpt_status = "✅ Connected"
            else:
                gpt_status = f"❌ Error {test_response.status_code}"
        except Exception as e:
            gpt_status = f"❌ Not reachable"
        
        log_info(f"System status requested")
        
        return jsonify({
            "status": "running",
            "embedding_model": "Azure OpenAI text-embedding-3-small",
            "llm_model": "Azure OpenAI GPT-4o-mini",
            "gpt_status": gpt_status,
            "total_chunks": total_chunks,
            "rag_feature": "✅ RAG Only - No Memory",
            "chromadb_path": os.path.abspath(config.CHROMA_PATH),
            "upload_folder": os.path.abspath(config.UPLOAD_FOLDER)
        })
    except Exception as e:
        log_error(f"Error getting system status", error=str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_everything():
    try:
        log_info("Clear all data requested")
        
        global collection, client
        client = PersistentClient(path=config.CHROMA_PATH)
        try:
            client.delete_collection(name="documents")
            log_info("Deleted ChromaDB collection")
        except:
            pass
        
        collection = client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine", "dimension": config.EMBEDDING_DIMENSION}
        )
        
        log_info("Cleared all data")
        
        return jsonify({
            "success": True,
            "message": "All documents cleared successfully"
        })
    except Exception as e:
        log_error(f"Error clearing data", error=str(e))
        return jsonify({"error": str(e)}), 500

# ==================== FRONTEND SERVING ====================
@app.route('/')
def serve_index():
    return send_from_directory(config.FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    return send_from_directory(config.FRONTEND_DIR, filename)

# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 60)
    print("📚 RAG SYSTEM ONLY (NO MEMORY)")
    print("=" * 60)
    print(f"📁 Upload folder: {os.path.abspath(config.UPLOAD_FOLDER)}")
    print(f"💾 ChromaDB path: {os.path.abspath(config.CHROMA_PATH)}")
    print(f"🔗 Embeddings: Azure OpenAI")
    print(f"🤖 LLM: Azure OpenAI GPT-4o-mini")
    print(f"📄 Chunking: PARAGRAPH-BASED with overlap")
    print("=" * 60)
    print("✅ RAG FEATURES ONLY:")
    print("  • Document upload (PDF, DOCX, TXT)")
    print("  • Vector embeddings and search")
    print("  • Document-based Q&A")
    print("  • No conversation memory")
    print("=" * 60)
    print("🌐 Open: http://localhost:5001")
    print("=" * 60)
    
    app.run(debug=True, port=5001, use_reloader=False)