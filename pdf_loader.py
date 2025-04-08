# pdf_loader.py
# This module handles the loading and processing of PDF documents

import os
import io
import tempfile
from typing import List, Dict, Any

import fitz  # PyMuPDF
from langchain_core.documents import Document


class PDFLoader:
    """Handles loading and processing of PDF documents"""

    def __init__(self):
        """Initialize the PDF loader"""
        pass

    def load_pdf(self, pdf_file) -> List[Document]:
        """
        Extract text and metadata from a PDF file
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            List of Document objects with text content and metadata
        """
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name

        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(temp_path)
            
            # Extract text from each page
            documents = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Create a Document object with page content and metadata
                document = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_num + 1,
                        "total_pages": len(doc)
                    }
                )
                documents.append(document)
            
            return documents
        
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def extract_pdf_text(self, pdf_file) -> str:
        """
        Extract all text from a PDF file as a single string
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            String containing all text from the PDF
        """
        documents = self.load_pdf(pdf_file)
        return "\n\n".join([doc.page_content for doc in documents])