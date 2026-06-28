
import os
import json
import secrets
import uuid
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from chromadb import PersistentClient
import pypdf
from docx import Document
import requests
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
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
    MAX_MEMORY_MESSAGES = 20
    SESSION_TIMEOUT_MINUTES = 30
    
    # User Info
    USER_INFO_CONFIDENCE_THRESHOLD = 0.7
    MAX_USER_INFO_ENTRIES = 20
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate_config(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.CHROMA_PATH, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)

config = Config()
config.validate_config()

# ==================== SIMPLE LOGGING ====================
def setup_logging():
    """Setup simple but effective logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(config.LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log"),
        encoding='utf-8'
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def log_info(message: str, **kwargs):
    """Log info with context"""
    extra = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    if extra:
        logger.info(f"{message} | {extra}")
    else:
        logger.info(message)

def log_error(message: str, **kwargs):
    """Log error with context"""
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

# ==================== DYNAMIC USER INFO MANAGER ====================
class DynamicUserInfoManager:
    """Intelligent user info extraction with context and memory evolution"""
    
    def __init__(self):
        self.user_info_store = defaultdict(dict)  # session_id -> user_info
        self.conversation_context = defaultdict(list)  # session_id -> recent context
        self.info_confidence = defaultdict(dict)  # session_id -> field -> confidence
        
    def extract_info_from_message(self, session_id: str, message: str, role: str) -> Dict[str, Any]:
        """Extract user info from a single message with contextual analysis"""
        if role != "user":
            return {}
        
        message_lower = message.lower()
        extracted_info = {}
        confidence_scores = {}
        
        # ===== 1. DIRECT PATTERN MATCHING (High Confidence) =====
        patterns = {
            'name': [
                (r'my name is ([a-z\s]+)', 0.9),
                (r'i am ([a-z\s]+)', 0.8),
                (r'call me ([a-z\s]+)', 0.85),
                (r'you can call me ([a-z\s]+)', 0.8),
                (r'everyone calls me ([a-z\s]+)', 0.75),
                (r'people call me ([a-z\s]+)', 0.7),
                (r'my friends call me ([a-z\s]+)', 0.7)
            ],
            'location': [
                (r'i live in ([a-z\s,\-]+)', 0.85),
                (r'i am from ([a-z\s,\-]+)', 0.8),
                (r'i\'m from ([a-z\s,\-]+)', 0.8),
                (r'based in ([a-z\s,\-]+)', 0.75),
                (r'currently in ([a-z\s,\-]+)', 0.7),
                (r'located in ([a-z\s,\-]+)', 0.7),
                (r'my city is ([a-z\s,\-]+)', 0.8)
            ],
            'profession': [
                (r'i work as ([a-z\s]+)', 0.85),
                (r'i am a ([a-z\s]+)', 0.8),
                (r'my job is ([a-z\s]+)', 0.8),
                (r'i work at ([a-z\s]+)', 0.75),
                (r'i\'m employed at ([a-z\s]+)', 0.75),
                (r'my profession is ([a-z\s]+)', 0.85),
                (r'my occupation is ([a-z\s]+)', 0.85)
            ],
            'age': [
                (r'i am (\d+) years old', 0.9),
                (r'i\'m (\d+) years old', 0.9),
                (r'age (\d+)', 0.7),
                (r'turned (\d+)', 0.8),
                (r'my age is (\d+)', 0.85)
            ],
            'interests': [
                (r'i like ([a-z\s,]+)', 0.6),
                (r'i love ([a-z\s,]+)', 0.7),
                (r'i enjoy ([a-z\s,]+)', 0.65),
                (r'my hobby is ([a-z\s,]+)', 0.7),
                (r'i\'m interested in ([a-z\s,]+)', 0.65)
            ],
            'company': [
                (r'i work for ([a-z\s,\-]+)', 0.8),
                (r'i\'m at ([a-z\s,\-]+)', 0.7),
                (r'my company is ([a-z\s,\-]+)', 0.85),
                (r'the company i work for is ([a-z\s,\-]+)', 0.85)
            ]
        }
        
        for field, field_patterns in patterns.items():
            for pattern, confidence in field_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    value = match.group(1).strip().title()
                    if field in ['age'] and value.isdigit():
                        value = int(value)
                    extracted_info[field] = value
                    confidence_scores[field] = confidence
                    log_info(f"Direct pattern match", field=field, value=value, confidence=confidence)
                    break
        
        # ===== 2. CONTEXTUAL INFERENCE (Medium Confidence) =====
        current_context = self.conversation_context[session_id]
        
        # Infer profession from context
        if 'profession' not in extracted_info and current_context:
            context_text = ' '.join(current_context[-3:]).lower()
            profession_keywords = {
                'doctor': ['patient', 'hospital', 'clinic', 'medicine', 'surgery'],
                'engineer': ['code', 'software', 'developer', 'programming', 'tech'],
                'teacher': ['student', 'class', 'school', 'teach', 'education'],
                'lawyer': ['case', 'court', 'legal', 'law', 'client'],
                'manager': ['team', 'project', 'manage', 'lead', 'report']
            }
            
            for profession, keywords in profession_keywords.items():
                if any(keyword in context_text for keyword in keywords):
                    extracted_info['profession'] = profession.title()
                    confidence_scores['profession'] = 0.6
                    log_info(f"Contextual inference", field='profession', value=profession, confidence=0.6)
                    break
        
        # Infer location from context
        if 'location' not in extracted_info and current_context:
            location_pattern = re.compile(r'\b(in|from|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE)
            for context in current_context[-2:]:
                matches = location_pattern.findall(context)
                if matches:
                    for _, location in matches:
                        if len(location.split()) <= 3 and location.lower() not in ['the', 'a', 'an', 'my', 'your']:
                            extracted_info['inferred_location'] = location
                            confidence_scores['inferred_location'] = 0.5
                            break
        
        # ===== 3. ENTITY RECOGNITION (Using simple patterns) =====
        # Name recognition from capitalization patterns
        if 'name' not in extracted_info:
            name_candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', message)
            for candidate in name_candidates:
                words = candidate.split()
                if len(words) <= 3 and len(words[0]) > 2:
                    # Check if it's likely a name (not common words)
                    common_words = ['the', 'and', 'but', 'for', 'with', 'this', 'that', 'have']
                    if candidate.lower() not in common_words:
                        extracted_info['possible_name'] = candidate
                        confidence_scores['possible_name'] = 0.4
                        break
        
        # ===== 4. UPDATE CONFIDENCE AND STORE =====
        existing_info = self.user_info_store.get(session_id, {})
        existing_confidence = self.info_confidence.get(session_id, {})
        
        for field, value in extracted_info.items():
            current_confidence = confidence_scores.get(field, 0.5)
            
            # If we already have this info, decide whether to update
            if field in existing_info:
                existing_confidence_level = existing_confidence.get(field, 0)
                
                # Update only if new info has higher confidence
                if current_confidence > existing_confidence_level:
                    existing_info[field] = value
                    existing_confidence[field] = current_confidence
                    log_info(f"Updated user info", field=field, value=value, 
                            old_confidence=existing_confidence_level, new_confidence=current_confidence)
            else:
                # New info, add it
                existing_info[field] = value
                existing_confidence[field] = current_confidence
                log_info(f"Added new user info", field=field, value=value, confidence=current_confidence)
        
        # Store updated info
        self.user_info_store[session_id] = existing_info
        self.info_confidence[session_id] = existing_confidence
        
        # Update conversation context
        self.conversation_context[session_id].append(message)
        if len(self.conversation_context[session_id]) > 10:
            self.conversation_context[session_id] = self.conversation_context[session_id][-10:]
        
        return extracted_info
    
    def infer_from_conversation_flow(self, session_id: str, conversation: List[Dict]) -> Dict[str, Any]:
        """Infer user info from overall conversation flow"""
        inferred_info = {}
        
        # Get all user messages
        user_messages = [msg['content'] for msg in conversation if msg.get('role') == 'user']
        if not user_messages:
            return inferred_info
        
        full_conversation = ' '.join(user_messages).lower()
        
        # Analyze conversation topics
        topic_keywords = {
            'technology': ['computer', 'code', 'software', 'app', 'website', 'tech', 'programming'],
            'business': ['meeting', 'client', 'project', 'deadline', 'company', 'business'],
            'education': ['study', 'school', 'university', 'course', 'learn', 'teacher'],
            'health': ['health', 'doctor', 'hospital', 'medicine', 'fitness', 'exercise']
        }
        
        for topic, keywords in topic_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in full_conversation)
            if keyword_count >= 2:
                inferred_info[f'interested_in_{topic}'] = True
        
        # Relationship detection
        relationship_indicators = [
            ('family', ['my wife', 'my husband', 'my children', 'my kids', 'my parents']),
            ('friends', ['my friends', 'my buddy', 'my colleague', 'my coworker']),
            ('pets', ['my dog', 'my cat', 'my pet'])
        ]
        
        for relationship, indicators in relationship_indicators:
            if any(indicator in full_conversation for indicator in indicators):
                inferred_info[f'has_{relationship}'] = True
        
        return inferred_info
    
    def get_user_info_summary(self, session_id: str) -> Dict[str, Any]:
        """Get structured user info with confidence scores"""
        info = self.user_info_store.get(session_id, {})
        confidence = self.info_confidence.get(session_id, {})
        
        # Filter only high-confidence info
        high_confidence_info = {}
        for field, value in info.items():
            if confidence.get(field, 0) >= config.USER_INFO_CONFIDENCE_THRESHOLD:
                high_confidence_info[field] = value
        
        return {
            'info': high_confidence_info,
            'all_info': info,
            'confidence_scores': confidence,
            'info_count': len(high_confidence_info),
            'has_name': 'name' in high_confidence_info,
            'has_location': any(k in high_confidence_info for k in ['location', 'inferred_location']),
            'has_profession': 'profession' in high_confidence_info
        }
    
    def clear_user_info(self, session_id: str):
        """Clear all user info for a session"""
        if session_id in self.user_info_store:
            del self.user_info_store[session_id]
        if session_id in self.info_confidence:
            del self.info_confidence[session_id]
        if session_id in self.conversation_context:
            del self.conversation_context[session_id]
        log_info(f"Cleared user info", session_id=session_id[:8])
    
    def evolve_user_info(self, session_id: str, new_conversation: List[Dict]):
        """Evolve user info based on new conversation"""
        for msg in new_conversation:
            if msg.get('role') == 'user':
                self.extract_info_from_message(session_id, msg['content'], 'user')
        
        # Add inferences from conversation flow
        inferred_info = self.infer_from_conversation_flow(session_id, new_conversation)
        for key, value in inferred_info.items():
            if key not in self.user_info_store.get(session_id, {}):
                self.user_info_store[session_id][key] = value
                self.info_confidence[session_id][key] = 0.5  # Medium confidence for inferences

# ==================== LANGGRAPH STATE ====================
class GraphState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    user_info: Dict[str, Any]
    user_info_summary: Dict[str, Any]
    session_id: str
    query: str
    document_context: str
    relevant_chunks: List[str]
    answer: str
    metadata: Dict[str, Any]

# ==================== MEMORY MANAGER (LANGGRAPH) ====================
class MemoryManager:
    """LangGraph-based memory and session management with DYNAMIC user info"""
    
    def __init__(self):
        self.memory_store = {}
        self.session_timeout = config.SESSION_TIMEOUT_MINUTES
        self.user_info_manager = DynamicUserInfoManager()
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        log_info("MemoryManager initialized with dynamic user info")
    
    def _build_graph(self):
        """Build LangGraph workflow for conversation processing"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("extract_user_info", self._extract_user_info)
        workflow.add_node("evolve_user_info", self._evolve_user_info)
        workflow.add_node("check_document_context", self._check_document_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_memory", self._update_memory)
        
        # Add edges
        workflow.set_entry_point("extract_user_info")
        workflow.add_edge("extract_user_info", "evolve_user_info")
        workflow.add_edge("evolve_user_info", "check_document_context")
        workflow.add_edge("check_document_context", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow.compile()
    
    def _extract_user_info(self, state: GraphState) -> GraphState:
        """Extract user info from current query"""
        session_id = state.get("session_id")
        query = state.get("query", "")
        
        if session_id and query:
            # Extract info from current query
            extracted = self.user_info_manager.extract_info_from_message(
                session_id, query, "user"
            )
            
            if extracted:
                log_info(f"Extracted info from query", session_id=session_id[:8], 
                        info_count=len(extracted))
        
        return state
    
    def _evolve_user_info(self, state: GraphState) -> GraphState:
        """Evolve user info based on entire conversation"""
        session_id = state.get("session_id")
        messages = state.get("messages", [])
        
        if session_id:
            # Evolve user info from conversation
            self.user_info_manager.evolve_user_info(session_id, messages)
            
            # Get updated user info summary
            user_info_summary = self.user_info_manager.get_user_info_summary(session_id)
            state["user_info_summary"] = user_info_summary
            
            # Prepare user info for prompt
            info_for_prompt = {}
            for field, value in user_info_summary.get('info', {}).items():
                if user_info_summary['confidence_scores'].get(field, 0) >= 0.6:
                    info_for_prompt[field] = value
            
            state["user_info"] = info_for_prompt
            
            if info_for_prompt:
                log_info(f"User info evolved", session_id=session_id[:8], 
                        info_count=len(info_for_prompt))
        
        return state
    
    def _check_document_context(self, state: GraphState) -> GraphState:
        """Check if document context is available for the query"""
        has_document_context = bool(
            state.get("document_context") and 
            len(state.get("document_context", "").strip()) > 100
        )
        
        state["metadata"] = state.get("metadata", {})
        state["metadata"]["has_document_context"] = has_document_context
        
        return state
    
    def _generate_response(self, state: GraphState) -> GraphState:
        """Generate AI response with personalized context"""
        # Prepare messages for GPT
        messages = []
        
        # DYNAMIC PERSONALIZED system prompt
        user_info = state.get("user_info", {})
        user_info_summary = state.get("user_info_summary", {})
        
        personalization_context = ""
        if user_info:
            personalization_context = "PERSONALIZED CONTEXT (From our conversation):\n"
            for key, value in user_info.items():
                personalization_context += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            # Add confidence note
            if user_info_summary.get('confidence_scores'):
                personalization_context += f"\nNote: I've learned these details about you from our conversation.\n"
        
        if state.get("metadata", {}).get("has_document_context", False):
            system_prompt = f"""You are a helpful, personalized assistant that remembers user details.

{personalization_context}

DOCUMENT CONTEXT:
{state.get('document_context', '')}

INSTRUCTIONS:
1. Answer based on document context when possible
2. Use and reference user information naturally in responses
3. If user asks about themselves, use the personalized context above
4. Be conversational and remember previous interactions
5. If answer is not in document, say so politely but still personalize response"""
        else:
            system_prompt = f"""You are a helpful assistant with memory of conversations and user details.

{personalization_context}

INSTRUCTIONS:
1. Use conversation history for context
2. Naturally incorporate user information in responses
3. If user asks about themselves, use the personalized context above
4. Be conversational and remember previous interactions
5. Ask clarifying questions if needed to learn more about the user"""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history (last 5 messages)
        recent_messages = state.get("messages", [])[-5:]
        for msg in recent_messages:
            if isinstance(msg, dict):
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        # Add current query
        messages.append({"role": "user", "content": state["query"]})
        
        # Call GPT
        answer = call_gpt(messages, max_tokens=500)
        
        if answer:
            state["answer"] = answer
            log_info("Generated personalized response", 
                    response_length=len(answer),
                    user_info_used=len(user_info))
        else:
            state["answer"] = "I apologize, but I couldn't generate a response. Please try again."
        
        return state
    
    def _update_memory(self, state: GraphState) -> GraphState:
        """Update conversation memory with new messages"""
        # Add user query to messages
        if state.get("query"):
            state["messages"].append({
                "role": "user",
                "content": state["query"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Add assistant response to messages
        if state.get("answer"):
            state["messages"].append({
                "role": "assistant",
                "content": state["answer"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Trim messages if too many
        if len(state["messages"]) > config.MAX_MEMORY_MESSAGES:
            state["messages"] = state["messages"][-config.MAX_MEMORY_MESSAGES:]
        
        return state
    
    def process_query_sync(self, session_id: str, query: str, 
                          document_context: str = "", 
                          relevant_chunks: List[str] = None) -> Dict[str, Any]:
        """Process a query through the LangGraph workflow"""
        log_info("Processing query with dynamic user info", 
                session_id=session_id[:8], query=query[:100])
        
        # Get or initialize session state
        session_state = self._get_session_state(session_id)
        
        # Ensure messages are in proper dict format
        messages = []
        for msg in session_state.get("messages", []):
            if isinstance(msg, dict):
                messages.append(msg)
        
        # Prepare initial state
        initial_state = GraphState(
            messages=messages,
            user_info={},
            user_info_summary={},
            session_id=session_id,
            query=query,
            document_context=document_context or "",
            relevant_chunks=relevant_chunks or [],
            answer="",
            metadata={}
        )
        
        try:
            # Execute graph
            result = self.graph.invoke(initial_state)
            
            # Update session state
            self._update_session_state(session_id, {
                "messages": result["messages"][-config.MAX_MEMORY_MESSAGES:],
                "user_info": result.get("user_info", {}),
                "last_activity": datetime.now()
            })
            
            return {
                "answer": result.get("answer", ""),
                "user_info": result.get("user_info", {}),
                "user_info_summary": result.get("user_info_summary", {}),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            log_error("Error processing query", session_id=session_id[:8], error=str(e))
            raise
    
    def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get session state from memory store"""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = {
                "messages": [],
                "user_info": {},
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
            log_info("Created new session", session_id=session_id[:8])
        else:
            # Check for timeout
            last_activity = self.memory_store[session_id]["last_activity"]
            if datetime.now() - last_activity > timedelta(minutes=self.session_timeout):
                log_info("Session expired, creating new", session_id=session_id[:8])
                self.memory_store[session_id] = {
                    "messages": [],
                    "user_info": {},
                    "created_at": datetime.now(),
                    "last_activity": datetime.now()
                }
                # Also clear user info for this session
                self.user_info_manager.clear_user_info(session_id)
            else:
                self.memory_store[session_id]["last_activity"] = datetime.now()
        
        return self.memory_store[session_id]
    
    def _update_session_state(self, session_id: str, updates: Dict[str, Any]):
        """Update session state in memory store"""
        if session_id in self.memory_store:
            self.memory_store[session_id].update(updates)
        else:
            self.memory_store[session_id] = {
                "messages": updates.get("messages", []),
                "user_info": updates.get("user_info", {}),
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information with user info"""
        if session_id not in self.memory_store:
            return None
        
        session_data = self.memory_store[session_id]
        session_age = datetime.now() - session_data["created_at"]
        
        # Get user info summary
        user_info_summary = self.user_info_manager.get_user_info_summary(session_id)
        
        return {
            "session_id": session_id[:8],
            "message_count": len(session_data["messages"]),
            "user_info": user_info_summary.get("info", {}),
            "user_info_count": user_info_summary.get("info_count", 0),
            "has_name": user_info_summary.get("has_name", False),
            "has_location": user_info_summary.get("has_location", False),
            "has_profession": user_info_summary.get("has_profession", False),
            "created_at": session_data["created_at"].isoformat(),
            "last_activity": session_data["last_activity"].isoformat(),
            "session_age_minutes": int(session_age.total_seconds() / 60),
            "is_active": (datetime.now() - session_data["last_activity"]).seconds < 300
        }
    
    def clear_session(self, session_id: str):
        """Clear session memory and user info"""
        if session_id in self.memory_store:
            self.memory_store[session_id] = {
                "messages": [],
                "user_info": {},
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
            # Also clear dynamic user info
            self.user_info_manager.clear_user_info(session_id)
            log_info("Session cleared with user info", session_id=session_id[:8])
            return True
        return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions with user info"""
        sessions = []
        for session_id, data in self.memory_store.items():
            user_info_summary = self.user_info_manager.get_user_info_summary(session_id)
            session_age = datetime.now() - data["created_at"]
            sessions.append({
                "session_id": session_id[:8],
                "message_count": len(data["messages"]),
                "user_info_count": user_info_summary.get("info_count", 0),
                "has_name": user_info_summary.get("has_name", False),
                "has_location": user_info_summary.get("has_location", False),
                "has_profession": user_info_summary.get("has_profession", False),
                "created_at": data["created_at"].isoformat(),
                "age_minutes": int(session_age.total_seconds() / 60)
            })
        return sessions

# Initialize Memory Manager
memory_manager = MemoryManager()

# ==================== DOCUMENT PROCESSING ====================
class DocumentProcessor:
    """Handles document reading and intelligent paragraph-based chunking"""
    
    @staticmethod
    def clean_document_text(text: str) -> str:
        """Clean document text by removing formatting artifacts"""
        if not text:
            return ""
        
        # Remove common formatting artifacts
        text = re.sub(r'\{#[^}]+\}', '', text)
        text = re.sub(r'\.[A-Za-z\-]+\}', '', text)
        text = re.sub(r'\{\.TOC-Heading\}', '', text)
        
        # Split into lines and clean
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
        """Read text from PDF file"""
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
        """Read text from DOCX file"""
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
        """Read text from TXT file"""
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
        """
        PARAGRAPH-BASED chunking with context preservation
        Returns list of chunks with metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        log_info("Starting paragraph-based chunking", total_paragraphs=len(paragraphs))
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 1
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_words = paragraph.split()
            para_length = len(paragraph)
            
            # Check if paragraph is too large
            if para_length > max_chunk_size:
                # If we have accumulated content, save it first
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': {
                            'chunk_id': chunk_id,
                            'chunk_type': 'paragraph_group',
                            'paragraph_range': f"{para_idx - len(current_chunk)}-{para_idx-1}",
                            'paragraph_count': len(current_chunk),
                            'total_paragraphs': len(paragraphs)
                        }
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                # Split the large paragraph into sentences
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
                                'chunk_type': 'sentence_group',
                                'source_paragraph': para_idx,
                                'sentence_count': len(sentence_chunk),
                                'total_paragraphs': len(paragraphs)
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
                            'chunk_type': 'sentence_group',
                            'source_paragraph': para_idx,
                            'sentence_count': len(sentence_chunk),
                            'total_paragraphs': len(paragraphs)
                        }
                    })
                    chunk_id += 1
            
            else:
                # Normal paragraph - add to current chunk
                if current_length + para_length > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': {
                            'chunk_id': chunk_id,
                            'chunk_type': 'paragraph_group',
                            'paragraph_range': f"{para_idx - len(current_chunk)}-{para_idx-1}",
                            'paragraph_count': len(current_chunk),
                            'total_paragraphs': len(paragraphs)
                        }
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_paras = current_chunk[-overlap:] if overlap > 0 else []
                    current_chunk = overlap_paras + para_words
                    current_length = len(' '.join(current_chunk))
                else:
                    current_chunk.extend(para_words)
                    current_length += para_length + 1
        
        # Add the last chunk if exists
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': {
                    'chunk_id': chunk_id,
                    'chunk_type': 'paragraph_group',
                    'paragraph_range': f"{len(paragraphs) - len(current_chunk)}-{len(paragraphs)-1}",
                    'paragraph_count': len(current_chunk),
                    'total_paragraphs': len(paragraphs)
                }
            })
        
        log_info("Paragraph chunking completed", total_chunks=len(chunks))
        
        return chunks
    
    @staticmethod
    def process_file(file_path: str, filename: str) -> tuple:
        """Process any supported file type"""
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
        
        # PARAGRAPH-BASED chunking
        chunks = DocumentProcessor.chunk_by_paragraphs(text)
        
        return chunks, text

# ==================== AZURE OPENAI FUNCTIONS ====================
def get_azure_embedding(text: str, max_retries: int = 3):
    """Get embeddings from Azure OpenAI"""
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
    """Call GPT-4o-mini via Azure OpenAI"""
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
    """Upload and process document"""
    log_info("File upload requested")
    
    if 'file' not in request.files:
        log_error("No file in request")
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        log_error("Empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    log_info(f"Processing file upload: {file.filename}")
    
    # Save file temporarily
    file_path = os.path.join(config.UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        # Process document
        chunks, full_text = DocumentProcessor.process_file(file_path, file.filename)
        
        if not chunks:
            raise ValueError("No content chunks created from file")
        
        log_info(f"Document processed", filename=file.filename, total_chunks=len(chunks))
        
        # Store chunks in ChromaDB
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
                    # Add metadata from document processor
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
    """Main chat endpoint with dynamic user info"""
    query = request.args.get('query', '').strip()
    if not query:
        log_error("Empty chat query")
        return jsonify({"error": "Please enter a question"}), 400
    
    # Get or create session
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
        session.permanent = True
        log_info(f"New session created: {session['session_id'][:8]}")
    
    session_id = session['session_id']
    
    log_info(f"Chat query received", session_id=session_id[:8], query=query[:100])
    
    # Search for relevant document chunks
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
    
    # Process query through LangGraph memory manager
    try:
        result = memory_manager.process_query_sync(
            session_id=session_id,
            query=query,
            document_context=document_context,
            relevant_chunks=relevant_chunks
        )
        
        answer = result.get("answer", "I couldn't generate a response.")
        metadata = result.get("metadata", {})
        user_info = result.get("user_info", {})
        user_info_summary = result.get("user_info_summary", {})
        
        # Prepare response
        response_data = {
            "answer": answer,
            "session_id": session_id[:8],
            "relevant_chunks": len(relevant_chunks),
            "found_in_document": metadata.get("has_document_context", False),
            "user_info_available": bool(user_info),
            "user_info": user_info,
            "user_info_summary": user_info_summary,
            "personalized": len(user_info) > 0
        }
        
        log_info(f"Response generated with dynamic user info", 
                answer_length=len(answer),
                user_info_count=len(user_info))
        
        return jsonify(response_data)
        
    except Exception as e:
        log_error(f"Error processing chat query", error=str(e))
        return jsonify({
            "error": "An error occurred while processing your query",
            "answer": "I'm sorry, but I encountered an error. Please try again."
        }), 500

@app.route('/session/new', methods=['GET'])
def new_session():
    """Create a new session"""
    old_session_id = session.get('session_id')
    
    # Create new session
    session.clear()
    session['session_id'] = secrets.token_hex(16)
    session.permanent = True
    
    new_session_id = session['session_id']
    
    log_info(f"New session created", new_session=new_session_id[:8], old_session=old_session_id[:8] if old_session_id else 'none')
    
    return jsonify({
        "success": True,
        "message": "New session created",
        "session_id": new_session_id[:8]
    })

@app.route('/session/clear', methods=['GET'])
def clear_session():
    """Clear current session memory"""
    if 'session_id' not in session:
        log_error("No active session to clear")
        return jsonify({"error": "No active session"}), 400
    
    session_id = session['session_id']
    
    success = memory_manager.clear_session(session_id)
    
    if success:
        log_info(f"Session memory cleared", session_id=session_id[:8])
        return jsonify({
            "success": True,
            "message": "Session memory cleared",
            "session_id": session_id[:8]
        })
    else:
        return jsonify({"error": "Session not found"}), 404

@app.route('/session/conversation', methods=['GET'])
def get_conversation():
    """Get full conversation history"""
    if 'session_id' not in session:
        return jsonify({"error": "No active session"}), 400
    
    session_id = session['session_id']
    session_state = memory_manager._get_session_state(session_id)
    
    log_info(f"Conversation history requested", session_id=session_id[:8])
    
    return jsonify({
        "success": True,
        "session_id": session_id[:8],
        "conversation": session_state.get("messages", []),
        "message_count": len(session_state.get("messages", [])),
        "user_info": memory_manager.user_info_manager.get_user_info_summary(session_id)
    })

@app.route('/session/info', methods=['GET'])
def get_session_info():
    """Get current session information with dynamic user info"""
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

@app.route('/session/userinfo', methods=['GET'])
def get_user_info():
    """Get detailed user information for current session"""
    if 'session_id' not in session:
        return jsonify({"error": "No active session"}), 400
    
    session_id = session['session_id']
    user_info_summary = memory_manager.user_info_manager.get_user_info_summary(session_id)
    
    return jsonify({
        "success": True,
        "session_id": session_id[:8],
        "user_info": user_info_summary.get("info", {}),
        "all_info": user_info_summary.get("all_info", {}),
        "confidence_scores": user_info_summary.get("confidence_scores", {}),
        "info_count": user_info_summary.get("info_count", 0),
        "has_name": user_info_summary.get("has_name", False),
        "has_location": user_info_summary.get("has_location", False),
        "has_profession": user_info_summary.get("has_profession", False)
    })

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
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
    """Get system status with user info stats"""
    try:
        results = collection.get()
        total_chunks = len(results['ids']) if results and results['ids'] else 0
        
        # Test GPT connection
        gpt_status = "Not tested"
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
            gpt_status = f"❌ Not reachable: {str(e)}"
        
        # Get active sessions with user info
        all_sessions = memory_manager.get_all_sessions()
        sessions_with_info = sum(1 for s in all_sessions if s['user_info_count'] > 0)
        
        log_info(f"System status requested")
        
        return jsonify({
            "status": "running",
            "embedding_model": "Azure OpenAI text-embedding-3-small",
            "llm_model": "Azure OpenAI GPT-4o-mini",
            "gpt_status": gpt_status,
            "total_chunks": total_chunks,
            "active_sessions": len(all_sessions),
            "sessions_with_user_info": sessions_with_info,
            "user_info_feature": "✅ Dynamic Extraction Enabled",
            "chromadb_path": os.path.abspath(config.CHROMA_PATH),
            "upload_folder": os.path.abspath(config.UPLOAD_FOLDER)
        })
    except Exception as e:
        log_error(f"Error getting system status", error=str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_everything():
    """Clear all data - documents and sessions"""
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
        memory_manager.memory_store = {}
        
        # Clear all user info
        memory_manager.user_info_manager = DynamicUserInfoManager()
        
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

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions with user info"""
    try:
        sessions = memory_manager.get_all_sessions()
        log_info(f"Sessions list requested", session_count=len(sessions))
        return jsonify({
            "success": True,
            "sessions": sessions,
            "total_sessions": len(sessions),
            "sessions_with_user_info": sum(1 for s in sessions if s['user_info_count'] > 0)
        })
    except Exception as e:
        log_error(f"Error listing sessions", error=str(e))
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
    print("🚀 COMPLETE RAG CHATBOT WITH DYNAMIC USER INFO")
    print("=" * 60)
    print(f"📁 Upload folder: {os.path.abspath(config.UPLOAD_FOLDER)}")
    print(f"💾 ChromaDB path: {os.path.abspath(config.CHROMA_PATH)}")
    print(f"📊 Logs directory: {os.path.abspath(config.LOGS_DIR)}")
    print(f"🔗 Embeddings: Azure OpenAI")
    print(f"🤖 LLM: Azure OpenAI GPT-4o-mini")
    print(f"🧠 Memory: LangGraph-based")
    print(f"👤 User Info: DYNAMIC EXTRACTION (Name, Location, Profession, etc.)")
    print(f"📄 Chunking: PARAGRAPH-BASED with overlap")
    print("=" * 60)
    print("✅ ALL FEATURES ENABLED:")
    print("  • Dynamic User Info Extraction from conversations")
    print("  • Confidence-based information storage")
    print("  • Contextual inference (not just pattern matching)")
    print("  • Memory evolution across conversations")
    print("  • Personalized responses using extracted info")
    print("  • Paragraph-based chunking")
    print("  • Full LangGraph workflow")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)
    
    # Test GPT connection
    print("\n🔗 Testing GPT-4o-mini connection...")
    try:
        test_response = requests.post(
            f"{config.GPT_ENDPOINT}?api-version={config.GPT_API_VERSION}",
            headers={"api-key": config.GPT_API_KEY, "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": "Hello, are you working?"}], "max_tokens": 20},
            timeout=10
        )
        if test_response.status_code == 200:
            print("✅ GPT-4o-mini: Connected and working!")
        else:
            print(f"⚠️ GPT-4o-mini: Error {test_response.status_code}")
    except Exception as e:
        print(f"❌ GPT-4o-mini: Connection failed - {e}")
    
    # Start Flask app
    app.run(debug=True, port=5000, use_reloader=False)