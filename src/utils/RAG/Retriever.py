import os, tempfile, re, numpy as np, logging
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import bm25s
from sklearn.metrics.pairwise import cosine_similarity

from utils.RAG.Processor import MedicalTextProcessor

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, text_processor: MedicalTextProcessor, embeddings: HuggingFaceEmbeddings):
        self.text_processor = text_processor
        self.embeddings = embeddings
        self.bm25_corpus = []
        self.bm25_model = None
    
    def initialize_bm25(self, documents: List[str]):
        self.bm25_corpus = documents
        if documents:
            corpus_tokens = bm25s.tokenize(documents, stopwords="en")
            self.bm25_model = bm25s.BM25()
            self.bm25_model.index(corpus_tokens)
        else:
            self.bm25_model = None
    
    def bm25_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.bm25_model or not self.bm25_corpus:
            return []

        query_tokens = bm25s.tokenize(query, stopwords="en")

        results, scores = self.bm25_model.retrieve(query_tokens, k=min(k, len(self.bm25_corpus)))
        
        bm25_results = []
        for i in range(results.shape[1]):
            doc_text = results[0, i]
            # Ensure we're creating proper Document objects
            bm25_results.append(Document(page_content=str(doc_text)))
        
        return bm25_results
    

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents
            
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Extract content safely and create embeddings
            doc_contents = []
            doc_embeddings = []
            
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict) and 'page_content' in doc:
                    content = doc['page_content']
                else:
                    content = str(doc)
                
                doc_contents.append(content)
                doc_embeddings.append(self.embeddings.embed_query(content))
            
            semantic_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            if self.bm25_model and self.bm25_corpus:
                # Get BM25 scores for reranking
                query_tokens = bm25s.tokenize(query, stopwords="en")
                
                # Get scores for all documents in corpus
                all_results, all_scores = self.bm25_model.retrieve(
                    query_tokens, 
                    k=len(self.bm25_corpus)
                )
                
                # Create a mapping of document content to BM25 scores
                doc_to_score = {}
                for i in range(all_results.shape[1]):
                    doc_text = all_results[0, i]
                    score = all_scores[0, i]
                    doc_to_score[doc_text] = score
                
                # Get BM25 scores for current documents
                bm25_scores = []
                for content in doc_contents:
                    bm25_scores.append(doc_to_score.get(content, 0.0))
                
                # Normalize BM25 scores
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
                bm25_scores = [score/max_bm25 for score in bm25_scores]
                
                # Combine scores
                combined_scores = [
                    0.6 * semantic_scores[i] + 0.4 * bm25_scores[i] 
                    for i in range(len(documents))
                ]
            else:
                combined_scores = semantic_scores

            scored_docs = list(zip(documents, combined_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs[:5]]  
            
        except Exception as e:
            logger.warning(f"Re-ranking failed, using original order: {e}")
            return documents[:5]


    def parallel_hybrid_retrieve(self, query: str, expanded_query: str, 
                                vector_results: List[Document], k: int = 5) -> List[Document]:
        bm25_results = self.bm25_search(query, k)
        
        all_candidates = vector_results + bm25_results
        seen = set()
        unique_docs = []
        
        for doc in all_candidates:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict) and 'page_content' in doc:
                content = doc['page_content']
            else:
                content = str(doc)
                
            h = hash(content[:500])   
            if h not in seen:
                seen.add(h)
                # Ensure we're returning proper Document objects
                if not isinstance(doc, Document):
                    doc = Document(page_content=content)
                unique_docs.append(doc)
        
        reranked = self.rerank_documents(query, unique_docs)
        return reranked[:k]