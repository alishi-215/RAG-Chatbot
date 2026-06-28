import os
import json
import secrets
import uuid
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from chromadb import PersistentClient
import pypdf
from docx import Document
import requests
import logging
import sys
from collections import defaultdict

# ==================== CONFIGURATION ====================
class Config:
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = ""
    AZURE_API_KEY = ""
    API_VERSION = ""
    
    GPT_ENDPOINT = ""
    GPT_API_KEY = ""
    GPT_API_VERSION = ""
    
    # Model Name
    GPT_MODEL = ""
    
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
    
    # Memory
    MAX_MEMORY_MESSAGES = 100
    SESSION_TIMEOUT_MINUTES = 30
    
    # User Info
    USER_INFO_CONFIDENCE_THRESHOLD = 0.7
    
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
        os.path.join(config.LOGS_DIR, f"complete_v3_{datetime.now().strftime('%Y%m%d')}.log"),
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
app.config['PERMANENT_SESSION_LIFETIME'] = config.SESSION_TIMEOUT_MINUTES * 60

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

# ==================== IMPROVED MEMORY SYSTEM ====================
class AdvancedMemoryManager:
    """Advanced memory system that remembers EVERYTHING"""
    
    def __init__(self):
        self.conversation_store = defaultdict(list)  # session_id -> list of messages
        self.user_info_store = defaultdict(dict)     # session_id -> user info
        self.session_timestamps = defaultdict(dict)  # session_id -> created_at, last_activity
        self.message_counter = defaultdict(int)      # session_id -> message_count
        
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_number": self.message_counter[session_id] + 1
        }
        self.conversation_store[session_id].append(message)
        self.message_counter[session_id] += 1
        
        # Trim if too many messages
        if len(self.conversation_store[session_id]) > config.MAX_MEMORY_MESSAGES:
            self.conversation_store[session_id] = self.conversation_store[session_id][-config.MAX_MEMORY_MESSAGES:]
        
        # Initialize timestamps if this is the first message
        if session_id not in self.session_timestamps:
            self.session_timestamps[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
        else:
            # Update last activity for existing sessions
            self.session_timestamps[session_id]["last_activity"] = datetime.now()
        
        # Extract user info from message
        if role == "user":
            self._extract_user_info(session_id, content)
    
    def _extract_user_info(self, session_id: str, message: str):
        """Extract user information from messages"""
        message_lower = message.lower()
        
        patterns = {
            'name': [
                (r'my name is ([a-z\s]+?)(?:\.|\!|\?|$)', 0.9),  # Fixed pattern
                (r'i am ([a-z\s]+?)(?:\.|\!|\?|$)', 0.8),        # Fixed pattern
                (r'call me ([a-z\s]+?)(?:\.|\!|\?|$)', 0.85),
                (r'you can call me ([a-z\s]+?)(?:\.|\!|\?|$)', 0.8),
            ],
            'location': [
                (r'i live in ([a-z\s,\-]+?)(?:\.|\!|\?|$)', 0.85),
                (r'i am from ([a-z\s,\-]+?)(?:\.|\!|\?|$)', 0.8),
            ],
            'profession': [
                (r'i work as ([a-z\s]+?)(?:\.|\!|\?|$)', 0.85),
                (r'i am a ([a-z\s]+?)(?:\.|\!|\?|$)', 0.8),
            ],
            'date_info': [
                (r'today(?:\'s| is) (\d+\s+[a-z]+\s+\d{4})', 0.9),
                (r'date is (\d+\s+[a-z]+\s+\d{4})', 0.85),
                (r'today\'s date is (\d+\s+[a-z]+\s+\d{4})', 0.95),
                (r'today date is (\d+\s+[a-z]+\s+\d{4})', 0.9),  # Fixed pattern
            ]
        }
        
        for field, field_patterns in patterns.items():
            for pattern, confidence in field_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    try:
                        if field == 'date_info':
                            # Check which group has the date
                            if match.lastindex and match.lastindex >= 2:
                                value = match.group(2).strip().title()
                            else:
                                value = match.group(1).strip().title()
                            self.user_info_store[session_id]['current_date'] = value
                            log_info(f"Extracted date info", session_id=session_id[:8], date=value)
                        else:
                            value = match.group(1).strip().title()
                            self.user_info_store[session_id][field] = value
                            log_info(f"Extracted user info", session_id=session_id[:8], field=field, value=value)
                        break
                    except IndexError as e:
                        log_error(f"Error extracting user info for pattern {pattern}", 
                                 error=str(e), message=message_lower)
                        continue
                    except Exception as e:
                        log_error(f"Unexpected error extracting user info", 
                                 error=str(e), pattern=pattern)
                        continue
    
    def get_conversation_history(self, session_id: str, max_messages: int = None) -> str:
        """Get formatted conversation history"""
        if session_id not in self.conversation_store:
            return "No previous conversation."
        
        if max_messages:
            messages = self.conversation_store[session_id][-max_messages:]
        else:
            messages = self.conversation_store[session_id]
        
        history_text = ""
        for msg in messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            msg_num = msg.get("message_number", 0)
            history_text += f"Message #{msg_num} - {role}: {content}\n"
        
        return history_text
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get summary of conversation"""
        if session_id not in self.conversation_store:
            return "No conversation yet."
        
        messages = self.conversation_store[session_id]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        summary = f"CONVERSATION SUMMARY:\n"
        summary += f"Total messages: {len(messages)}\n"
        summary += f"User messages: {len(user_messages)}\n"
        summary += f"Assistant messages: {len(assistant_messages)}\n"
        
        if user_messages:
            summary += f"\nUSER QUESTIONS ({len(user_messages)} total):\n"
            for i, msg in enumerate(user_messages, 1):
                summary += f"{i}. {msg.get('content', '')[:100]}{'...' if len(msg.get('content', '')) > 100 else ''}\n"
        
        return summary
    
    def count_user_questions(self, session_id: str) -> int:
        """Count how many questions user has asked"""
        if session_id not in self.conversation_store:
            return 0
        
        user_messages = [msg for msg in self.conversation_store[session_id] 
                        if msg.get("role") == "user"]
        return len(user_messages)
    
    def get_all_user_questions(self, session_id: str) -> List[str]:
        """Get all questions asked by user"""
        if session_id not in self.conversation_store:
            return []
        
        user_messages = [msg for msg in self.conversation_store[session_id] 
                        if msg.get("role") == "user"]
        return [msg.get("content", "") for msg in user_messages]
    
    def get_user_info_summary(self, session_id: str) -> str:
        """Get formatted user info summary"""
        if session_id not in self.user_info_store or not self.user_info_store[session_id]:
            return "No user information available yet."
        
        info = self.user_info_store[session_id]
        summary = "USER INFORMATION (from conversation):\n"
        for key, value in info.items():
            summary += f"- {key.replace('_', ' ').title()}: {value}\n"
        return summary
    
    def clear_session(self, session_id: str):
        """Clear session data"""
        if session_id in self.conversation_store:
            del self.conversation_store[session_id]
        if session_id in self.user_info_store:
            del self.user_info_store[session_id]
        if session_id in self.session_timestamps:
            del self.session_timestamps[session_id]
        if session_id in self.message_counter:
            del self.message_counter[session_id]
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        if session_id not in self.conversation_store:
            return None
        
        # Check if session has timestamps, if not create them
        if session_id not in self.session_timestamps:
            self.session_timestamps[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
        
        session_data = self.session_timestamps[session_id]
        
        # Ensure created_at exists
        if "created_at" not in session_data:
            session_data["created_at"] = datetime.now()
        
        # Ensure last_activity exists
        if "last_activity" not in session_data:
            session_data["last_activity"] = datetime.now()
        
        session_age = datetime.now() - session_data["created_at"]
        
        return {
            "session_id": session_id[:8],
            "message_count": len(self.conversation_store.get(session_id, [])),
            "user_question_count": self.count_user_questions(session_id),
            "user_info": self.user_info_store.get(session_id, {}),
            "user_info_count": len(self.user_info_store.get(session_id, {})),
            "created_at": session_data["created_at"].isoformat(),
            "last_activity": session_data["last_activity"].isoformat(),
            "session_age_minutes": int(session_age.total_seconds() / 60)
        }

# Initialize Memory Manager
memory_manager = AdvancedMemoryManager()

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

def call_gpt(messages: List[Dict], max_tokens: int = 800):
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
    
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
        session.permanent = True
        log_info(f"New session created: {session['session_id'][:8]}")
    
    session_id = session['session_id']
    
    log_info(f"Chat query", session_id=session_id[:8], query=query[:100])
    
    # First, add user query to memory
    try:
        memory_manager.add_message(session_id, "user", query)
    except Exception as e:
        log_error(f"Error adding message to memory", error=str(e))
    
    # Search for relevant document chunks
    document_context = ""
    query_embedding = get_azure_embedding(query)
    source_documents = []  # NEW: List to store source document names
    
    if query_embedding:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            if results and 'documents' in results and results['documents']:
                relevant_chunks = results['documents'][0] if results['documents'][0] else []
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                
                if relevant_chunks:
                    log_info(f"Found relevant document chunks", chunk_count=len(relevant_chunks))
                    
                    context_parts = []
                    for i, chunk in enumerate(relevant_chunks):
                        context_parts.append(f"[Document chunk {i+1}]:\n{chunk}")
                        
                        # NEW: Extract document name from metadata
                        if i < len(metadatas) and metadatas[i]:
                            filename = metadatas[i].get('filename', 'Unknown Document')
                            if filename not in source_documents:
                                source_documents.append(filename)
                    
                    document_context = "\n\n---\n\n".join(context_parts)
        except Exception as e:
            log_error(f"Error searching ChromaDB", error=str(e))
    
    try:
        # Get COMPLETE conversation history
        conversation_history = memory_manager.get_conversation_history(session_id)
        conversation_summary = memory_manager.get_conversation_summary(session_id)
        user_info_summary = memory_manager.get_user_info_summary(session_id)
        
        # Count user questions
        user_question_count = memory_manager.count_user_questions(session_id)
        all_user_questions = memory_manager.get_all_user_questions(session_id)
        
        # Prepare messages for GPT
        messages = []
        
        # Build COMPREHENSIVE system prompt
        system_prompt = f"""You are an intelligent assistant with COMPLETE MEMORY of the entire conversation.

==================== CURRENT CONTEXT ====================
CURRENT DATE: {datetime.now().strftime("%d %B %Y")}
CURRENT TIME: {datetime.now().strftime("%H:%M:%S")}
SESSION ID: {session_id[:8]}
TOTAL MESSAGES IN SESSION: {len(memory_manager.conversation_store.get(session_id, []))}
USER QUESTIONS COUNT: {user_question_count}

==================== USER INFORMATION ====================
{user_info_summary}

==================== CONVERSATION SUMMARY ====================
{conversation_summary}

==================== COMPLETE CONVERSATION HISTORY ====================
{conversation_history}

==================== DOCUMENT CONTEXT ====================
{document_context if document_context else "No specific document context for this query."}

==================== CRITICAL INSTRUCTIONS ====================
1. You have COMPLETE MEMORY of the entire conversation above
2. When user asks about conversation history, REFER to the exact messages
3. When counting questions, use the exact count: {user_question_count} user questions
4. NEVER say "I don't remember" or "You haven't asked" - you have full history
5. If user asks "what questions did I ask?", list them from history
6. If user asks "how many questions?", answer: {user_question_count}
7. If user asks about their name, check user information section
8. Current date is: {datetime.now().strftime("%d %B %Y")}
9. Always acknowledge that you remember the conversation
10. For date-related questions, use current date unless user specified otherwise

==================== NOW RESPOND TO USER ==================="""

        messages.append({"role": "system", "content": system_prompt})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Call GPT with more tokens for comprehensive responses
        answer = call_gpt(messages, max_tokens=1000)
        
        if answer:
            # Add assistant response to memory
            memory_manager.add_message(session_id, "assistant", answer)
            
            # Prepare response
            from_memory = True
            found_in_document = bool(document_context)
            
            response_data = {
                "answer": answer,
                "session_id": session_id[:8],
                "user_info_available": bool(memory_manager.user_info_store.get(session_id, {})),
                "user_info": memory_manager.user_info_store.get(session_id, {}),
                "message_count": len(memory_manager.conversation_store.get(session_id, [])),
                "user_question_count": user_question_count,
                "from_memory": from_memory,
                "found_in_document": found_in_document,
                # NEW: Add model name and source documents to response
                "model_name": config.GPT_MODEL,
                "source_documents": source_documents,  # This contains document names
                "source_document_count": len(source_documents)
            }
            
            log_info(f"Response generated", 
                    answer_length=len(answer),
                    user_question_count=user_question_count,
                    source_documents=source_documents)
            
            return jsonify(response_data)
        else:
            return jsonify({"error": "Failed to generate response"}), 500
        
    except Exception as e:
        log_error(f"Error processing chat query", error=str(e))
        return jsonify({
            "error": "An error occurred while processing your query",
            "answer": "I'm sorry, but I encountered an error. Please try again."
        }), 500

@app.route('/session/new', methods=['GET'])
def new_session():
    old_session_id = session.get('session_id')
    
    if old_session_id:
        memory_manager.clear_session(old_session_id)
        log_info(f"Cleared old session", session_id=old_session_id[:8])
    
    session.clear()
    session['session_id'] = secrets.token_hex(16)
    session.permanent = True
    
    new_session_id = session['session_id']
    log_info(f"New session created", new_session=new_session_id[:8])
    
    return jsonify({
        "success": True,
        "message": "New session created",
        "session_id": new_session_id[:8]
    })

@app.route('/session/clear', methods=['GET'])
def clear_session():
    if 'session_id' not in session:
        return jsonify({"error": "No active session"}), 400
    
    session_id = session['session_id']
    memory_manager.clear_session(session_id)
    
    log_info(f"Session memory cleared", session_id=session_id[:8])
    return jsonify({
        "success": True,
        "message": "Session memory cleared",
        "session_id": session_id[:8]
    })

@app.route('/session/conversation', methods=['GET'])
def get_conversation():
    if 'session_id' not in session:
        return jsonify({"error": "No active session"}), 400
    
    session_id = session['session_id']
    conversation = memory_manager.conversation_store.get(session_id, [])
    
    log_info(f"Conversation history requested", 
            session_id=session_id[:8], 
            message_count=len(conversation))
    
    return jsonify({
        "success": True,
        "session_id": session_id[:8],
        "conversation": conversation,
        "message_count": len(conversation),
        "user_question_count": memory_manager.count_user_questions(session_id),
        "user_info": memory_manager.user_info_store.get(session_id, {})
    })

@app.route('/session/info', methods=['GET'])
def get_session_info():
    if 'session_id' not in session:
        return jsonify({"has_session": False})
    
    session_id = session['session_id']
    session_info = memory_manager.get_session_info(session_id)
    
    if session_info:
        return jsonify({
            "has_session": True,
            **session_info
        })
    else:
        return jsonify({"has_session": False})

@app.route('/session/questions', methods=['GET'])
def get_user_questions():
    if 'session_id' not in session:
        return jsonify({"error": "No active session"}), 400
    
    session_id = session['session_id']
    questions = memory_manager.get_all_user_questions(session_id)
    
    return jsonify({
        "success": True,
        "session_id": session_id[:8],
        "questions": questions,
        "question_count": len(questions)
    })

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
        
        # Count active sessions
        active_sessions = len(memory_manager.conversation_store)
        
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
            "llm_model": config.GPT_MODEL,
            "gpt_status": gpt_status,
            "total_chunks": total_chunks,
            "active_sessions": active_sessions,
            "memory_feature": "✅ COMPLETE MEMORY (100 messages)",
            "rag_feature": "✅ RAG Enabled",
            "upload_feature": "✅ Upload Enabled",
            "document_source_feature": "✅ Shows Document Names",
            "current_date": datetime.now().strftime("%d %B %Y"),
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
        
        # Clear ChromaDB collection
        global collection, client
        client = PersistentClient(path=config.CHROMA_PATH)
        try:
            client.delete_collection(name="documents")
            log_info("Deleted ChromaDB collection")
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine", "dimension": config.EMBEDDING_DIMENSION}
        )
        
        # Clear all sessions
        memory_manager.conversation_store.clear()
        memory_manager.user_info_store.clear()
        memory_manager.session_timestamps.clear()
        memory_manager.message_counter.clear()
        
        # Clear current session
        session.clear()
        
        log_info("Cleared all data")
        
        return jsonify({
            "success": True,
            "message": "All data cleared successfully"
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
    print("🚀 RAG CHATBOT WITH COMPLETE MEMORY & DOCUMENT SOURCES")
    print("=" * 60)
    print(f"📁 Upload folder: {os.path.abspath(config.UPLOAD_FOLDER)}")
    print(f"💾 ChromaDB path: {os.path.abspath(config.CHROMA_PATH)}")
    print(f"🧠 Memory: COMPLETE (stores {config.MAX_MEMORY_MESSAGES} messages)")
    print(f"📄 Document Sources: SHOWS DOCUMENT NAMES")
    print(f"🤖 Model: {config.GPT_MODEL}")
    print("=" * 60)
    print("✅ ENHANCED FEATURES:")
    print("  • Shows exact document names for answers")
    print("  • Remembers ALL conversation history")
    print("  • Tracks user question count")
    print("  • Stores user information")
    print("  • Can list all user questions")
    print("  • Knows exact message numbers")
    print("  • Maintains conversation context")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000, use_reloader=False)