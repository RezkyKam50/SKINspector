import os, tempfile, re, numpy as np, logging
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MedicalTextProcessor:
    # TODO: Extend abbreviations from UMLS
    def __init__(self):
        self.medical_abbreviations = {
            'BCC': 'Basal Cell Carcinoma',
            'SCC': 'Squamous Cell Carcinoma',
            'AK': 'Actinic Keratosis',
            'MM': 'Malignant Melanoma',
            'TPO': 'Treatment Plan and Options'
        }
    
    def tokenize_medical_text(self, text: str) -> List[str]:
        text = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1\2', text)  
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())  
        return tokens
    
    def clean_medical_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'/L\d+', '', text)   # remove artifact refs
        text = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', text)  # fix split words

        for abbr, full in self.medical_abbreviations.items():
            text = text.replace(f' {abbr} ', f' {full} ({abbr}) ')
            
        return text.strip()
    
    def clean_context_text(self, content: str) -> str:
        cleaned_content = re.sub(r'\[\d+\]', '', content)   
        cleaned_content = re.sub(r'\b\d+\.?\d*\b', '', cleaned_content)  
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        return cleaned_content



class DocumentProcessor:
    def __init__(self, text_processor: MedicalTextProcessor):
        self.text_processor = text_processor
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,   
            chunk_overlap=150,
            separators=["\n\n## ", "\n\n", "\n", ". ", "! ", "? ", " ", ""],   
            length_function=len,
        )
    
    def process_document(self, file_path: str) -> List[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            cleaned_documents = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    cleaned_content = self.text_processor.clean_medical_text(doc.page_content)
                    cleaned_doc = Document(
                        page_content=cleaned_content,
                        metadata=doc.metadata
                    )
                    cleaned_documents.append(cleaned_doc)
                else:
                    cleaned_documents.append(doc)
            
            chunks = self.text_splitter.split_documents(cleaned_documents)
            logger.info(f"Processed {len(documents)} pages into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise