# embedder.py
# This module handles text embedding and vector storage

import os
from typing import List, Dict, Any, Optional
import pickle

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class DocumentEmbedder:
    """Handles document chunking, embedding, and vector storage"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 vector_store_type: str = "faiss"):
        """
        Initialize the document embedder
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model to use
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
            vector_store_type: Type of vector store ("faiss" or "chroma")
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store_type = vector_store_type
        self.vector_store = None
        
        # Create directory for storing embeddings
        os.makedirs("embeddings", exist_ok=True)
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Document]) -> Any:
        """
        Create and return a vector store from documents
        
        Args:
            documents: List of Document objects (chunks)
            
        Returns:
            Vector store object (FAISS or Chroma)
        """
        # Split documents into chunks if they haven't been split already
        chunks = self.process_documents(documents)
        
        # Create vector store
        if self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        elif self.vector_store_type.lower() == "chroma":
            self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        return self.vector_store
    
    def save_vector_store(self, pdf_name: str) -> None:
        """
        Save the vector store to disk
        
        Args:
            pdf_name: Name of the PDF file (for creating filename)
        """
        if self.vector_store is None:
            raise ValueError("No vector store available to save")
        
        # Create a safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in pdf_name)
        save_path = os.path.join("embeddings", safe_name)
        
        if self.vector_store_type.lower() == "faiss":
            self.vector_store.save_local(save_path)
        elif self.vector_store_type.lower() == "chroma":
            # For Chroma, we need to use the persist method
            # This is just a placeholder as the actual implementation depends on specific use case
            pass
    
    def load_vector_store(self, pdf_name: str) -> Optional[Any]:
        """
        Load a vector store from disk
        
        Args:
            pdf_name: Name of the PDF file
            
        Returns:
            Vector store object or None if not found
        """
        # Create a safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in pdf_name)
        load_path = os.path.join("embeddings", safe_name)
        
        if os.path.exists(load_path):
            if self.vector_store_type.lower() == "faiss":
                self.vector_store = FAISS.load_local(load_path, self.embeddings)
                return self.vector_store
        
        return None
    
    def get_retriever(self, search_type: str = "mmr", k: int = 5):
        """
        Create a retriever from the vector store
        
        Args:
            search_type: Type of search to perform ("mmr" or "similarity")
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("No vector store available for retrieval")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )