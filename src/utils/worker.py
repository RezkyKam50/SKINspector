from utils.base64_encoder import image_to_base64_data_uri
from utils.RAG.RAG import DermatologyRAGSystem

from PyQt6.QtCore import pyqtSignal, QObject, QThread

import os, logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    text_chunk = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)

    def __init__(self, llm_vl, llm, file_path, rag_system: DermatologyRAGSystem = None, use_rag: bool = False):
        super().__init__()
        self.llm_vl = llm_vl
        self.llm = llm
        self.file_path = file_path
        self.rag_system = rag_system
        self.use_rag = use_rag
        self._is_running = True

    def stop(self):
        self._is_running = False

    def process_image(self):
        try:
            if not self.file_path or not self._is_running:
                return
            
            self.progress_update.emit(10, "Processing image...")
            data_uri = image_to_base64_data_uri(self.file_path)

            if self.use_rag and self.rag_system:
                self.progress_update.emit(30, "Searching medical knowledge base...")
                messages = self.rag_system.build_rag_messages(data_uri)
                self.progress_update.emit(50, "Medical context loaded, starting analysis...")
            else:
                messages = [
                    {
                        "role": "system", 
                        "content": f"""You are an expert dermatologist. Provide a detailed analysis including:
                        1. Potential conditions that match the presentation
                        2. Key characteristics observed
                        Use International English language."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": "Analyze this skin condition. Describe what you see and provide potential scenarios."}
                        ]
                    }
                ]

            if self._is_running:
                response = self.llm_vl.create_chat_completion(
                    messages=messages, 
                    stream=True,

                    repeat_penalty=1.0,     
                    temperature=0.7,    

                    top_p=0.75,              
                    top_k=14,      
                )
                for event in response:
                    if not self._is_running:
                        break
                    if "choices" in event:
                        delta = event["choices"][0]["delta"]
                        if "content" in delta:
                            self.text_chunk.emit(delta["content"])

            if self._is_running:
                self.progress_update.emit(100, "Analysis complete!")
                self.finished.emit()

        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            self.error.emit(str(e))



class DocumentProcessingWorker(QObject):
    finished = pyqtSignal(int)  
    error = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)

    def __init__(self, rag_system: DermatologyRAGSystem, file_path: str):
        super().__init__()
        self.rag_system = rag_system
        self.file_path = file_path
        self._is_running = True

    def stop(self):
        self._is_running = False

    def process_document(self):
        try:
            if not self._is_running:
                return

            self.progress_update.emit(20, "Loading document...")
            chunks_added = self.rag_system.add_to_knowledge_base(self.file_path)
            
            if self._is_running:
                self.progress_update.emit(100, f"Added {chunks_added} knowledge chunks")
                self.finished.emit(chunks_added)

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            self.error.emit(str(e))

class StreamEmitter(QObject):
    new_text = pyqtSignal(str)

    def write(self, text):
        if text.strip():   
            self.new_text.emit(text)

    def flush(self):
        pass   



