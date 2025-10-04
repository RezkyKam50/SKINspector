# Author : Rezky M. Kam
# Do not use for real medical application. Only for reference.

import os, tempfile, re, numpy as np, logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from utils.RAG.Processor import MedicalTextProcessor, DocumentProcessor
from utils.RAG.Retriever import HybridRetriever
from utils.RAG.Prompting import MessageBuilder

logger = logging.getLogger(__name__)


class QueryExpander:
    def __init__(self):
        # TODO: Extend abbreviations from UMLS
        self.medical_synonyms = {
            "rash": "eruption dermatitis erythema",
            "acne": "comedones pimples pustules",
            "eczema": "atopic dermatitis inflammation",
            "psoriasis": "plaques scales",
            "melanoma": "skin cancer malignant",
            "lesion": "sore wound abnormality",
            "pruritus": "itching scratch",
            "erythema": "redness inflammation",
            "papule": "bump elevation",
            "pustule": "pus infection"
        }
    
    def expand_query(self, query: str) -> str:
        expanded_query = query.lower()
        for term, synonyms in self.medical_synonyms.items():
            if term in expanded_query:
                expanded_query += " " + synonyms
                
        return expanded_query



class DermatologyRAGSystem:    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/S-PubMedBert-MS-MARCO"  
        )
        
        self.text_processor = MedicalTextProcessor()
        self.query_expander = QueryExpander()
        self.hybrid_retriever = HybridRetriever(self.text_processor, self.embeddings)
        self.document_processor = DocumentProcessor(self.text_processor)
        self.message_builder = MessageBuilder(self.text_processor)
        
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing vector store")
                self._initialize_bm25()
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=[Document(page_content="Initial document")],
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                self.vectorstore.delete([self.vectorstore.get()['ids'][0]])
                logger.info("Created new vector store")
            
            self._setup_retriever()
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def _initialize_bm25(self):
        try:
            all_docs = self.vectorstore.get()
            documents = [doc for doc in all_docs.get('documents', [])]
            self.hybrid_retriever.initialize_bm25(documents)
        except Exception as e:
            logger.warning(f"Could not initialize BM25: {e}")

    def _tokenize_medical_text(self, text: str) -> List[str]:
        return self.text_processor.tokenize_medical_text(text)

    def _setup_retriever(self):
        try:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 12,  
                    "fetch_k": 20,
                    "lambda_mult": 0.5,  
                    "score_threshold": 0.5   
                }
            )
        except:
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )

    def _medical_query_expansion(self, query: str) -> str:
        return self.query_expander.expand_query(query)

    def _parallel_hybrid_retrieve(self, query: str, k: int = 5) -> List[Document]:
        expanded_query = self._medical_query_expansion(query)
        vector_results = self.retriever.invoke(expanded_query)
        return self.hybrid_retriever.parallel_hybrid_retrieve(
            query, expanded_query, vector_results, k
        )

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        return self.hybrid_retriever.rerank_documents(query, documents)

    def process_document(self, file_path: str) -> List[Document]:
        return self.document_processor.process_document(file_path)

    def _clean_medical_text(self, text: str) -> str:
        return self.text_processor.clean_medical_text(text)

    def add_to_knowledge_base(self, file_path: str) -> int:
        chunks = self.process_document(file_path)
        
        if chunks:
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
            self._initialize_bm25()
            self._setup_retriever() 
            logger.info(f"Added {len(chunks)} chunks to knowledge base")
        
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        try:
            stats = self.vectorstore.get()
            if not stats['ids'] or len(stats['ids']) == 0:
                logger.info("Knowledge base is empty, returning no results")
                return []
        except Exception as e:
            logger.warning(f"Could not check vectorstore stats: {e}")
            return []
        
        try:
            expanded_query = self._medical_query_expansion(query)
            logger.info(f"Original query: '{query}', Expanded: '{expanded_query}'")

            results = self.retriever.invoke(expanded_query)
            
            if not results:
                return []

            reranked_results = self._parallel_hybrid_retrieve(query, k)
            
            return reranked_results[:k] if reranked_results else []
            
        except Exception as e:
            logger.error(f"Error during enhanced retrieval: {e}")
            try:
                return self.retriever.invoke(query)[:k]
            except:
                return []

    def clear_knowledge_base(self):
        if self.vectorstore:
            all_ids = self.vectorstore.get()['ids']
            if all_ids:
                self.vectorstore.delete(all_ids)
                self.vectorstore.persist()
                logger.info("Cleared knowledge base")
            # Reset BM25
            self.hybrid_retriever.bm25_corpus = []
            self.hybrid_retriever.bm25_model = None
            # Reinitialize
            self._initialize_vectorstore()

    def get_stats(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"total_documents": 0, "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO"}
        
        try:
            collection_stats = self.vectorstore.get()
            total_docs = len(collection_stats['ids'])
            
            # Calculate average chunk length
            avg_length = 0
            if total_docs > 0 and 'documents' in collection_stats:
                doc_lengths = [len(doc) for doc in collection_stats['documents']]
                avg_length = sum(doc_lengths) / len(doc_lengths)
            
            return {
                "total_documents": total_docs,
                "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
                "average_chunk_length": round(avg_length, 2),
                "bm25_corpus_size": len(self.hybrid_retriever.bm25_corpus)
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"total_documents": 0, "embedding_model": "all-mpnet-base-v2"}

    def build_rag_messages(self, image_data_uri: str) -> List[Dict[str, Any]]:
        try:
            # Check if knowledge base has documents first
            stats = self.get_stats()
            if stats.get('total_documents', 0) == 0:
                logger.info("Knowledge base is empty, using fallback messages")
                return self._get_fallback_messages(image_data_uri)

            # TODO: Implememt auto-fill mech. using Classification model by top-n confidence score.
            search_queries = [
                "basal cell carcinoma",
                "skin lesions",
                "skin cancer",
                "melanoma",
                "squamous cell carcinoma",
                "actinic keratosis",
                "benign lesions",
                "malignant lesions",
                "non-melanoma skin cancer",
                "Bowen's disease",
                "keratoacanthoma",
                "seborrheic keratosis",
                "lentigo maligna",
                "dysplastic nevus",
                "atypical mole",
                "cutaneous lymphoma",
                "Merkel cell carcinoma",
                "dermatofibroma",
                "epidermoid cyst",
                "pyogenic granuloma",
                "angiokeratoma",
                "porokeratosis",
                "hidradenoma",
                "trichoepithelioma",
                "keratosis pilaris",
                "nevus sebaceous",
            ]

            all_relevant_docs = []
            for search_query in search_queries:
                docs = self.search(search_query, k=10)
                all_relevant_docs.extend(docs)

            seen_content = set()
            unique_docs = []
            for doc in all_relevant_docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                content_hash = hash(content.strip().lower())
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)

            return self.message_builder.build_rag_messages(image_data_uri, unique_docs)
            
        except Exception as e:
            logger.error(f"Error building enhanced RAG messages: {e}")
            return self._get_fallback_messages(image_data_uri)

    def _get_fallback_messages(self, image_data_uri: str) -> List[Dict[str, Any]]:
        return self.message_builder._get_fallback_messages(image_data_uri)