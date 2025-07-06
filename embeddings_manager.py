import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        """
        Initialize the embeddings manager with the specified model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self.embeddings_folder = "embeddings"
        self.relevance_threshold = 0.65  # Minimum similarity score for relevance
        self.dynamic_threshold = 0.45  # Lower threshold for dynamic responses
        
        # University-specific keywords for enhanced semantic understanding
        self.university_keywords = {
            'academic': ['course', 'program', 'degree', 'major', 'minor', 'curriculum', 'syllabus', 'academic', 'study'],
            'administrative': ['admission', 'enrollment', 'registration', 'application', 'deadline', 'procedure', 'process'],
            'financial': ['fee', 'tuition', 'payment', 'cost', 'scholarship', 'financial aid', 'budget', 'expense'],
            'campus': ['facility', 'building', 'campus', 'library', 'lab', 'classroom', 'dormitory', 'housing'],
            'student_life': ['student', 'life', 'activity', 'club', 'organization', 'event', 'campus life'],
            'services': ['service', 'support', 'help', 'assistance', 'guidance', 'counseling', 'advising']
        }
        
        # Create embeddings folder if it doesn't exist
        if not os.path.exists(self.embeddings_folder):
            os.makedirs(self.embeddings_folder)
    
    def create_embeddings(self, chunks):
        """
        Create embeddings for text chunks and build FAISS index
        
        Args:
            chunks: List of dictionaries with text and metadata
        """
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Create FAISS index - using IndexFlatIP for cosine similarity
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        return embeddings
    
    def save_embeddings(self, filename_prefix="university_combined"):
        """
        Save the embeddings and chunks to disk
        
        Args:
            filename_prefix: Prefix for the saved files
        """
        if self.index is None or self.chunks is None:
            raise ValueError("No embeddings or chunks to save")
        
        # Save the FAISS index
        index_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save the chunks
        chunks_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        return index_path, chunks_path
    
    def load_embeddings(self, filename_prefix="university_combined"):
        """
        Load embeddings and chunks from disk
        
        Args:
            filename_prefix: Prefix for the saved files
            
        Returns:
            True if successful, False otherwise
        """
        index_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_index.faiss")
        chunks_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_chunks.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return False
        
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the chunks
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        return True
    
    def combine_embeddings(self, sources):
        """
        Combine embeddings from multiple sources
        
        Args:
            sources: List of filename prefixes to combine
            
        Returns:
            True if successful, False otherwise
        """
        all_chunks = []
        
        # Load chunks from each source
        for source in sources:
            chunks_path = os.path.join(self.embeddings_folder, f"{source}_chunks.pkl")
            if not os.path.exists(chunks_path):
                return False
            
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
                all_chunks.extend(chunks)
        
        # Create new embeddings from combined chunks
        self.create_embeddings(all_chunks)
        
        # Save combined embeddings
        self.save_embeddings(filename_prefix="university_combined")
        
        return True
    
    def _extract_university_keywords(self, query: str) -> List[str]:
        """
        Extract university-related keywords from the query
        
        Args:
            query: User query text
            
        Returns:
            List of relevant university keywords found in the query
        """
        query_lower = query.lower()
        found_keywords = []
        
        for category, keywords in self.university_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_keywords.extend(keywords)
                    break
        
        return list(set(found_keywords))
    
    def _is_university_related(self, query: str) -> bool:
        """
        Check if the query is university-related
        
        Args:
            query: User query text
            
        Returns:
            True if the query is university-related
        """
        query_lower = query.lower()
        
        # Check for university-specific keywords
        for category, keywords in self.university_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return True
        
        # Check for common university terms
        university_terms = ['university', 'college', 'student', 'professor', 'lecturer', 'campus', 'academic']
        if any(term in query_lower for term in university_terms):
            return True
        
        return False
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance the query with related university terms for better semantic search
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with related terms
        """
        if not self._is_university_related(query):
            return query
        
        # Extract keywords and add related terms
        keywords = self._extract_university_keywords(query)
        enhanced_terms = []
        
        for keyword in keywords:
            # Find related terms from the same category
            for category, category_keywords in self.university_keywords.items():
                if keyword in category_keywords:
                    enhanced_terms.extend(category_keywords[:3])  # Add top 3 related terms
                    break
        
        # Combine original query with enhanced terms
        enhanced_query = query
        if enhanced_terms:
            enhanced_query += " " + " ".join(enhanced_terms[:5])  # Limit to 5 additional terms
        
        return enhanced_query
    
    def _smart_chunk_filtering(self, query: str, chunks: List[Dict], scores: List[float]) -> List[Dict]:
        """
        Smart filtering of chunks based on multiple criteria
        
        Args:
            query: User query
            chunks: List of chunks to filter
            scores: Similarity scores for each chunk
            
        Returns:
            Filtered list of relevant chunks
        """
        if not chunks:
            return []
        
        filtered_chunks = []
        query_lower = query.lower()
        
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            chunk_text_lower = chunk["text"].lower()
            
            # Check if this is a university-related query
            is_university_query = self._is_university_related(query)
            
            # Apply different filtering strategies based on query type
            if is_university_query:
                # For university queries, be more lenient with relevance scores
                if score >= self.dynamic_threshold:
                    # Additional semantic checks
                    keyword_match = any(keyword in chunk_text_lower 
                                      for keyword in self._extract_university_keywords(query))
                    
                    if keyword_match or score >= self.relevance_threshold:
                        chunk["metadata"]["relevance_score"] = float(score)
                        chunk["metadata"]["filtering_reason"] = "university_related"
                        filtered_chunks.append(chunk)
            else:
                # For non-university queries, use strict relevance threshold
                if score >= self.relevance_threshold:
                    chunk["metadata"]["relevance_score"] = float(score)
                    chunk["metadata"]["filtering_reason"] = "high_relevance"
                    filtered_chunks.append(chunk)
        
        # If no chunks passed the filter but we have results, return the best ones
        if not filtered_chunks and chunks:
            # Sort by score and return top 3
            sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            for chunk, score in sorted_chunks[:3]:
                chunk["metadata"]["relevance_score"] = float(score)
                chunk["metadata"]["filtering_reason"] = "fallback_best_match"
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def search_similar_chunks(self, query: str, k: int = 20) -> List[Dict]:
        """
        Enhanced search for chunks most similar to the query with intelligent fallback
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of relevant chunks with their metadata
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index or chunks not loaded")
        
        # Enhance the query for better semantic search
        enhanced_query = self._enhance_query(query)
        
        # Encode the enhanced query
        query_embedding = self.model.encode([enhanced_query], normalize_embeddings=True)
        
        # Search with larger k for better filtering
        search_k = min(k * 3, len(self.chunks))  # Search more chunks than needed
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Get the corresponding chunks
        results = []
        result_scores = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                result_scores.append(scores[0][i])
        
        # Apply smart filtering
        relevant_chunks = self._smart_chunk_filtering(query, results, result_scores)
        
        # If we don't have enough relevant chunks, try a broader search
        if len(relevant_chunks) < 3 and self._is_university_related(query):
            logger.info(f"Limited results for '{query}', trying broader search")
            broader_chunks = self._broad_search(query, k)
            if broader_chunks:
                relevant_chunks.extend(broader_chunks)
        
        # Remove duplicates and limit results
        seen_texts = set()
        final_chunks = []
        for chunk in relevant_chunks:
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                final_chunks.append(chunk)
                if len(final_chunks) >= k:
                    break
        
        return final_chunks
    
    def _broad_search(self, query: str, k: int) -> List[Dict]:
        """
        Perform a broader search when initial search yields limited results
        
        Args:
            query: User query
            k: Number of results to return
            
        Returns:
            List of additional relevant chunks
        """
        # Try searching with individual keywords
        keywords = self._extract_university_keywords(query)
        additional_chunks = []
        
        for keyword in keywords[:3]:  # Try top 3 keywords
            keyword_embedding = self.model.encode([keyword], normalize_embeddings=True)
            scores, indices = self.index.search(keyword_embedding, k)
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    score = scores[0][i]
                    
                    # Use lower threshold for keyword-based search
                    if score >= 0.4:  # Lower threshold for broader search
                        chunk["metadata"]["relevance_score"] = float(score)
                        chunk["metadata"]["filtering_reason"] = "keyword_search"
                        additional_chunks.append(chunk)
        
        return additional_chunks
    
    def get_chunks_by_source_type(self, source_type: str) -> List[Dict]:
        """
        Get chunks filtered by source type (pdf or web)
        
        Args:
            source_type: Type of source ("pdf" or "web")
            
        Returns:
            List of chunks from the specified source type
        """
        if not self.chunks:
            return []
        
        return [chunk for chunk in self.chunks if chunk["metadata"].get("type") == source_type]
    
    def get_chunks_by_category(self, category: str) -> List[Dict]:
        """
        Get chunks filtered by university category
        
        Args:
            category: University category (academic, administrative, financial, etc.)
            
        Returns:
            List of chunks from the specified category
        """
        if not self.chunks or category not in self.university_keywords:
            return []
        
        category_keywords = self.university_keywords[category]
        relevant_chunks = []
        
        for chunk in self.chunks:
            chunk_text_lower = chunk["text"].lower()
            if any(keyword in chunk_text_lower for keyword in category_keywords):
                relevant_chunks.append(chunk)
        
        return relevant_chunks 