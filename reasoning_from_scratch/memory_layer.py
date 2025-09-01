
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import numpy as np
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class SessionMemory:
    """Manages session memory for past reasoning interactions."""
    
    def __init__(self, max_sessions: int = 100):
        self.max_sessions = max_sessions
        self.sessions = []
        self.current_session = []
    
    def add_interaction(self, question: str, answer: str, reasoning: str, method: str):
        """Add an interaction to current session."""
        interaction = {
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "method": method,
            "timestamp": self._get_timestamp()
        }
        self.current_session.append(interaction)
    
    def end_session(self):
        """End current session and start a new one."""
        if self.current_session:
            self.sessions.append(self.current_session.copy())
            self.current_session = []
            
            # Keep only max_sessions
            if len(self.sessions) > self.max_sessions:
                self.sessions = self.sessions[-self.max_sessions:]
    
    def get_relevant_history(self, question: str, max_items: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant past interactions for the current question."""
        all_interactions = []
        
        # Add current session interactions
        all_interactions.extend(self.current_session)
        
        # Add interactions from past sessions
        for session in self.sessions:
            all_interactions.extend(session)
        
        if not all_interactions:
            return []
        
        # Simple relevance scoring based on keyword overlap
        scored_interactions = []
        question_words = set(question.lower().split())
        
        for interaction in all_interactions:
            interaction_words = set(interaction["question"].lower().split())
            overlap = len(question_words & interaction_words)
            scored_interactions.append((overlap, interaction))
        
        # Sort by relevance and return top items
        scored_interactions.sort(reverse=True)
        return [item[1] for item in scored_interactions[:max_items]]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def save(self, filepath: str):
        """Save session memory to file."""
        data = {
            "sessions": self.sessions,
            "current_session": self.current_session
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load session memory from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.sessions = data.get("sessions", [])
            self.current_session = data.get("current_session", [])
        except FileNotFoundError:
            pass


class DocumentRetriever:
    """RAG-based document retriever using embeddings and FAISS."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
        self.index = None
        self._setup_embedder()
    
    def _setup_embedder(self):
        """Setup the sentence embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("Warning: sentence-transformers not installed. Using dummy embedder.")
            self.embedder = None
    
    def add_documents(self, docs: List[str]):
        """Add documents to the retriever."""
        if not self.embedder:
            print("No embedder available. Documents added but not indexed.")
            self.documents.extend(docs)
            return
        
        # Generate embeddings
        new_embeddings = self.embedder.encode(docs)
        
        self.documents.extend(docs)
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index for fast similarity search."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(np.array(self.embeddings).astype('float32'))
        except ImportError:
            print("Warning: FAISS not installed. Using linear search.")
            self.index = None
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents for the query."""
        if not self.documents:
            return []
        
        if not self.embedder:
            # Fallback to simple keyword matching
            return self._keyword_retrieve(query, k)
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        
        if self.index:
            # Use FAISS for fast search
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                min(k, len(self.documents))
            )
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        "document": self.documents[idx],
                        "score": float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                        "rank": i + 1
                    })
            return results
        else:
            # Linear search fallback
            similarities = []
            query_emb = query_embedding[0]
            
            for i, doc_emb in enumerate(self.embeddings):
                # Cosine similarity
                dot_product = np.dot(query_emb, doc_emb)
                norm_product = np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                similarity = dot_product / norm_product if norm_product > 0 else 0
                similarities.append((similarity, i))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            results = []
            for rank, (sim, idx) in enumerate(similarities[:k]):
                results.append({
                    "document": self.documents[idx],
                    "score": float(sim),
                    "rank": rank + 1
                })
            
            return results
    
    def _keyword_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based retrieval."""
        query_words = set(query.lower().split())
        
        scored_docs = []
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scored_docs.append((overlap, i, doc))
        
        scored_docs.sort(reverse=True)
        
        results = []
        for rank, (score, idx, doc) in enumerate(scored_docs[:k]):
            results.append({
                "document": doc,
                "score": score / max(len(query_words), 1),
                "rank": rank + 1
            })
        
        return results
    
    def save_index(self, filepath: str):
        """Save the retriever state to files."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if len(self.embeddings) > 0 else []
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_index(self, filepath: str):
        """Load the retriever state from files."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.embeddings = np.array(data["embeddings"]) if data["embeddings"] else np.array([])
            
            if len(self.embeddings) > 0:
                self._build_index()
        except FileNotFoundError:
            pass


class KnowledgeMemoryLayer:
    """Combined knowledge and memory layer integrating RAG and session memory."""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.session_memory = SessionMemory()
        self.document_retriever = DocumentRetriever()
        
        # Load existing data
        self._load_state()
    
    def add_interaction(self, question: str, answer: str, reasoning: str, method: str):
        """Add a new reasoning interaction to memory."""
        self.session_memory.add_interaction(question, answer, reasoning, method)
        
        # Also add the reasoning as a document for future retrieval
        reasoning_doc = f"Question: {question}\nReasoning: {reasoning}\nAnswer: {answer}"
        self.document_retriever.add_documents([reasoning_doc])
    
    def add_knowledge_documents(self, documents: List[str]):
        """Add external knowledge documents."""
        self.document_retriever.add_documents(documents)
    
    def get_context_for_question(self, question: str, max_docs: int = 3, max_history: int = 2) -> Dict[str, Any]:
        """Get relevant context for answering a question."""
        # Retrieve relevant documents
        relevant_docs = self.document_retriever.retrieve(question, k=max_docs)
        
        # Get relevant history
        relevant_history = self.session_memory.get_relevant_history(question, max_items=max_history)
        
        return {
            "documents": relevant_docs,
            "history": relevant_history,
            "context_prompt": self._build_context_prompt(relevant_docs, relevant_history)
        }
    
    def _build_context_prompt(self, docs: List[Dict], history: List[Dict]) -> str:
        """Build a context prompt from retrieved documents and history."""
        context_parts = []
        
        if docs:
            context_parts.append("Relevant knowledge:")
            for i, doc in enumerate(docs[:3]):  # Limit to top 3
                context_parts.append(f"{i+1}. {doc['document'][:200]}...")
        
        if history:
            context_parts.append("\nPrevious related interactions:")
            for i, interaction in enumerate(history[:2]):  # Limit to top 2
                context_parts.append(
                    f"{i+1}. Q: {interaction['question'][:100]}...\n"
                    f"   A: {interaction['answer'][:100]}..."
                )
        
        return "\n".join(context_parts) if context_parts else ""
    
    def end_session(self):
        """End the current session."""
        self.session_memory.end_session()
        self._save_state()
    
    def _save_state(self):
        """Save the current state to disk."""
        # Save session memory
        session_file = self.memory_dir / "sessions.json"
        self.session_memory.save(str(session_file))
        
        # Save document index
        index_file = self.memory_dir / "document_index.pkl"
        self.document_retriever.save_index(str(index_file))
    
    def _load_state(self):
        """Load saved state from disk."""
        # Load session memory
        session_file = self.memory_dir / "sessions.json"
        if session_file.exists():
            self.session_memory.load(str(session_file))
        
        # Load document index
        index_file = self.memory_dir / "document_index.pkl"
        if index_file.exists():
            self.document_retriever.load_index(str(index_file))
