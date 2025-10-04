from typing import Optional, Dict
from utils.RAG.RAG import DermatologyRAGSystem
from utils.C_API import get_chat_vl, get_chat_llm


class SkinSpector:
    """Backend handler for SkinSpector application."""
    
    def __init__(self):
        """Initialize backend components."""
        _ , self.llm_vl = get_chat_vl()
        self.llm = None, #get_chat_llm()
        self.rag_system = DermatologyRAGSystem()
        self.file_path: Optional[str] = None
    
    def set_image_path(self, file_path: str) -> None:
        """Set the current image file path."""
        self.file_path = file_path
    
    def get_image_path(self) -> Optional[str]:
        """Get the current image file path."""
        return self.file_path
    
    def get_rag_system(self) -> DermatologyRAGSystem:
        """Get the RAG system instance."""
        return self.rag_system
    
    def get_llm_vl(self):
        """Get the LLM instance."""
        return self.llm_vl
    
    def get_llm(self):
        """Get the LLM instance."""
        return self.llm
    
    def get_kb_stats(self) -> Dict:
        """Get knowledge base statistics."""
        try:
            return self.rag_system.get_stats()
        except Exception as e:
            print(f"Error getting KB stats: {e}")
            return {"total_documents": 0, "embedding_model": "N/A"}
    
    def clear_knowledge_base(self) -> None:
        """Clear the knowledge base."""
        self.rag_system.clear_knowledge_base()
    
    def validate_image_file(self, file_path: str) -> bool:
        """Validate if the file is a supported image format."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        return file_path.lower().endswith(valid_extensions)
    
    def validate_document_file(self, file_path: str) -> bool:
        """Validate if the file is a supported document format."""
        valid_extensions = ('.pdf', '.docx', '.doc', '.txt')
        return file_path.lower().endswith(valid_extensions)