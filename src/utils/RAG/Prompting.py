from typing import List, Dict, Any, Optional
import os, tempfile, re, numpy as np, logging
 
from langchain_core.documents import Document
from utils.RAG.Processor import MedicalTextProcessor


logger = logging.getLogger(__name__)

class MessageBuilder:
    def __init__(self, text_processor: MedicalTextProcessor):
        self.text_processor = text_processor

    def build_rag_messages(self, image_data_uri: str, unique_docs: List[Document]) -> List[Dict[str, Any]]:
        try:
            if not unique_docs:
                logger.info("No relevant documents found, using fallback messages")
                return self._get_fallback_messages(image_data_uri)
            
            context_parts = []
            for doc in unique_docs[:3]:   
                try:
                    if isinstance(doc, Document):
                        content = doc.page_content
                    elif hasattr(doc, 'page_content'):
                        content = str(doc.page_content)
                    elif isinstance(doc, dict) and 'page_content' in doc:
                        content = doc['page_content']
                    elif isinstance(doc, str):
                        content = doc
                    else:
                        continue
                    
                    content = str(content).strip()
                    if content and len(content) > 50: 
                        cleaned_content = self.text_processor.clean_context_text(content)
                        
                        if len(cleaned_content) > 100:  
                            context_parts.append(cleaned_content)   
                            
                except Exception as e:
                    logger.warning(f"Error extracting content from document: {e}")
                    continue
            
            if context_parts:
                context_text = "\n\n".join(context_parts)   
                logger.info(f"Successfully built context from {len(context_parts)} documents")
            else:
                logger.warning("No valid context found, using fallback")
                return self._get_fallback_messages(image_data_uri)
            system_message = {
                "role": "system",
                "content": 
                f"""
                You are an expert dermatologist. Analyze the skin condition systematically and concisely.
                Here are some Medical References from textbooks:
                {context_text}
                Providing knowledge from above, you must answer concise, detailed and accurate answer. Do not repeat words, terms or acronyms.
                """
            }
            
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": "Describe the image."}
                ]
            }

            return [system_message, user_message]
            
        except Exception as e:
            logger.error(f"Error building RAG messages: {e}")
            return self._get_fallback_messages(image_data_uri)
        

    def _get_fallback_messages(self, image_data_uri: str) -> List[Dict[str, Any]]:
        system_message = {
            "role": "system",
            "content": (
                "You are an expert dermatologist working with the user whose also a dermatologist. Since no reference materials are available, "
                "analyze the skin condition purely from the provided image. "
                "Provide a concise, systematic, and accurate description of the condition, "
                "including potential diagnoses, severity, and recommended next steps. "
                "Provide answer to the user as a colleague."
                "Avoid speculation outside of visible evidence."
            ),
        }

        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": "Please describe the visible skin condition."},
            ],
        }

        return [system_message, user_message]
