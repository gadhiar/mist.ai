import sqlite3
import numpy as np
import json
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class RAGSystem:
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./embeddings_cache"):
        """
        Initialize the enhanced RAG system with embeddings.
        
        Args:
            db_path: Path to the SQLite database
            model_name: Name of the sentence-transformer model to use
            cache_dir: Directory to cache embeddings
        """
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Embedding model loaded on {self.device}")
        
        # Cache for embeddings
        self.embedding_cache = self._load_embedding_cache()
        
        # Initialize database tables if needed
        self._initialize_db()
        
        # Load existing conversations and cache embeddings
        self._load_conversations()
    
    def _initialize_db(self):
        """Create necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_message TEXT,
            ai_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create embeddings table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            text TEXT,
            embedding_file TEXT,
            chunk_id TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings from disk."""
        cache = {}
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                    for chunk_id, embedding_data in cache_data.items():
                        cache[chunk_id] = np.array(embedding_data)
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}")
        
        print(f"Loaded {len(cache)} cached embeddings")
        return cache
    
    def _save_embedding_to_cache(self, chunk_id: str, embedding: np.ndarray):
        """Save an embedding to the cache."""
        cache_file = self.cache_dir / f"{chunk_id[:2]}.json"
        
        # Load existing cache file or create new one
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
            except:
                cache_data = {}
        else:
            cache_data = {}
        
        # Add new embedding
        cache_data[chunk_id] = embedding.tolist()
        
        # Save cache file
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    
    def _load_conversations(self):
        """Load existing conversations from the database and cache their embeddings."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First check if the conversations table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
            if not cursor.fetchone():
                print("Conversations table does not exist yet")
                conn.close()
                return
            
            # Get all conversations not yet in the embeddings table
            cursor.execute("""
            SELECT c.id, c.user_message, c.ai_message 
            FROM conversations c
            LEFT JOIN embeddings e ON c.id = e.conversation_id
            WHERE e.id IS NULL
            """)
            
            conversations = cursor.fetchall()
            if not conversations:
                conn.close()
                return
            
            print(f"Processing embeddings for {len(conversations)} new conversations")
            
            for conv_id, user_message, ai_message in conversations:
                # Create chunks for indexing
                if user_message is not None and ai_message is not None:  # Ensure neither is None
                    self._process_conversation(conv_id, user_message, ai_message, conn)
            
            conn.close()
            print("Finished processing conversation embeddings")
        except Exception as e:
            print(f"Error loading conversations: {e}")
    
    def _process_conversation(self, conv_id: int, user_message: str, ai_message: str, conn: sqlite3.Connection):
        """Process a conversation and store its embeddings."""
        try:
            cursor = conn.cursor()
            
            # Combine user and AI messages
            full_text = f"User: {user_message}\nAI: {ai_message}"
            
            # For simplicity, we're using the entire message as a chunk
            # In a more sophisticated system, you might split long messages into smaller chunks
            chunk_id = f"conv_{conv_id}"
            
            # Check if we already have this embedding cached
            if chunk_id not in self.embedding_cache:
                # Generate embedding
                embedding = self.model.encode(full_text)
                self.embedding_cache[chunk_id] = embedding
                
                # Save to cache
                self._save_embedding_to_cache(chunk_id, embedding)
            else:
                embedding = self.embedding_cache[chunk_id]
            
            # Store reference in database
            embedding_file = f"{chunk_id[:2]}.json"
            cursor.execute(
                "INSERT INTO embeddings (conversation_id, text, embedding_file, chunk_id) VALUES (?, ?, ?, ?)",
                (conv_id, full_text, embedding_file, chunk_id)
            )
            
            conn.commit()
        except Exception as e:
            print(f"Error processing conversation {conv_id}: {e}")
    
    def add_conversation(self, user_message: str, ai_message: str) -> int:
        """
        Add a new conversation to the RAG system.
        
        Returns:
            The id of the newly added conversation
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert the conversation first
            cursor.execute(
                "INSERT INTO conversations (user_message, ai_message) VALUES (?, ?)",
                (user_message, ai_message)
            )
            
            # Get the ID of the new conversation
            conv_id = cursor.lastrowid
            
            # Process the conversation for embeddings
            self._process_conversation(conv_id, user_message, ai_message, conn)
            
            conn.commit()
            conn.close()
            
            return conv_id
        except Exception as e:
            print(f"Error adding conversation: {e}")
            return -1
    
    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Query the RAG system for relevant context.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            
        Returns:
            List of tuples with (text, similarity_score)
        """
        if query_text is None or query_text.strip() == "":
            print("Error: Empty query text")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query_text)
            
            # Find most similar conversations
            results = self._find_similar(query_embedding, top_k)
            
            return results
        except Exception as e:
            print(f"Error during query: {e}")
            return []
    
    def get_context(self, query_text: str, top_k: int = 3, threshold: float = 0.6) -> str:
        """
        Get formatted context from the RAG system.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            threshold: Minimum similarity score to include
            
        Returns:
            String with relevant context
        """
        results = self.query(query_text, top_k)
        
        if not results:
            return ""
        
        # Format results
        context_parts = []
        for text, score in results:
            # Only include really relevant results
            if score < threshold:  # Threshold for relevance
                continue
            context_parts.append(f"Relevance: {score:.2f}\n{text}")
        
        return "\n---\n".join(context_parts)
    
    def _find_similar(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Find the most similar conversations to the query."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the embeddings table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
            if not cursor.fetchone():
                print("Embeddings table does not exist yet")
                conn.close()
                return []
            
            # Get all embeddings from the database
            cursor.execute("SELECT id, text, embedding_file, chunk_id FROM embeddings")
            results = cursor.fetchall()
            
            if not results:
                conn.close()
                return []
            
            # Calculate similarities
            similarities = []
            for db_id, text, embedding_file, chunk_id in results:
                # Get embedding from cache
                if chunk_id in self.embedding_cache:
                    embedding = self.embedding_cache[chunk_id]
                else:
                    # This should not happen, but just in case
                    # We'll reload from the cache file
                    cache_file = self.cache_dir / embedding_file
                    if cache_file.exists():
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                            if chunk_id in cache_data:
                                embedding = np.array(cache_data[chunk_id])
                                self.embedding_cache[chunk_id] = embedding
                            else:
                                continue
                    else:
                        continue
                
                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((text, similarity))
            
            conn.close()
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            print(f"Error finding similar documents: {e}")
            return []
    
    def refresh(self):
        """Refresh embeddings from the database."""
        self._load_conversations()

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAGSystem("conversations.db")