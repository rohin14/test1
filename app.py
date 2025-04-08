# app.py
# Main Streamlit application file

import os
import json
import streamlit as st
from typing import Dict, List, Optional

from pdf_loader import PDFLoader
from embedder import DocumentEmbedder
from rag_pipeline import RAGPipeline

# Set page configuration
st.set_page_config(
    page_title="PDF Study Assistant",
    page_icon="📚",
    layout="wide"
)

# Constant for GROQ API Key (replace with your actual API key)
GROQ_API_KEY = "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"

# Initialize session state
def init_session_state():
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = {}  # Dictionary to store PDF documents
    
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = {}  # Dictionary to store vector stores
    
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None  # Currently selected PDF
    
    if "preview_text" not in st.session_state:
        st.session_state.preview_text = ""  # Preview text from the selected PDF
    
    if "pdf_loader" not in st.session_state:
        st.session_state.pdf_loader = PDFLoader()  # PDF loader instance
    
    if "embedder" not in st.session_state:
        st.session_state.embedder = DocumentEmbedder(
            embedding_model_name="all-MiniLM-L6-v2",
            vector_store_type="faiss"
        )
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"  # Change to your preferred model
        )

# Function to load PDFs
def load_pdfs(uploaded_files):
    for pdf_file in uploaded_files:
        if pdf_file.name not in st.session_state.pdf_docs:
            try:
                with st.spinner(f"Processing {pdf_file.name}..."):
                    # Load PDF and extract text
                    documents = st.session_state.pdf_loader.load_pdf(pdf_file)
                    
                    # Store the documents
                    st.session_state.pdf_docs[pdf_file.name] = documents
                    
                    # Create vector store
                    vector_store = st.session_state.embedder.create_vector_store(documents)
                    st.session_state.vector_stores[pdf_file.name] = vector_store
                    
                    # Save the vector store
                    st.session_state.embedder.save_vector_store(pdf_file.name)
                    
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing {pdf_file.name}: {str(e)}")

# Function to update the preview text
def update_preview(pdf_name):
    if pdf_name in st.session_state.pdf_docs:
        documents = st.session_state.pdf_docs[pdf_name]
        # Get text from the first few pages (up to 3)
        preview_pages = min(3, len(documents))
        preview_text = "\n\n".join([
            f"--- Page {documents[i].metadata['page']} ---\n{documents[i].page_content[:500]}..."
            for i in range(preview_pages)
        ])
        st.session_state.preview_text = preview_text
        st.session_state.current_pdf = pdf_name

# Function to process user query
def process_query(query, pdf_name):
    if pdf_name not in st.session_state.vector_stores:
        return "Please select a PDF first."
    
    try:
        # Get the vector store for the selected PDF
        vector_store = st.session_state.vector_stores[pdf_name]
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
        
        # Create QA chain
        qa_chain = st.session_state.rag_pipeline.create_qa_chain(retriever)
        
        # Get the answer
        with st.spinner("Generating answer..."):
            answer = qa_chain.invoke(query)
        
        return answer
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Function to generate summary
def generate_summary(pdf_name):
    if pdf_name not in st.session_state.vector_stores:
        return "Please select a PDF first."
    
    try:
        # Get the vector store for the selected PDF
        vector_store = st.session_state.vector_stores[pdf_name]
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10}  # Retrieve more chunks for summary
        )
        
        # Create summary chain
        summary_chain = st.session_state.rag_pipeline.create_summary_chain(retriever)
        
        # Get the summary
        with st.spinner("Generating summary..."):
            summary = summary_chain.invoke({"query": "summarize"})
        
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to generate study notes
def generate_notes(pdf_name):
    if pdf_name not in st.session_state.vector_stores:
        return "Please select a PDF first."
    
    try:
        # Get the vector store for the selected PDF
        vector_store = st.session_state.vector_stores[pdf_name]
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15}  # Retrieve more chunks for comprehensive notes
        )
        
        # Create notes chain
        notes_chain = st.session_state.rag_pipeline.create_notes_chain(retriever)
        
        # Get the notes
        with st.spinner("Generating study notes..."):
            notes = notes_chain.invoke({"query": "generate notes"})
        
        return notes
    
    except Exception as e:
        return f"Error generating notes: {str(e)}"

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Application title
    st.title("📚 PDF Study Assistant")
    st.markdown("**Upload PDFs, ask questions, generate summaries and study notes**")
    
    # Sidebar
    with st.sidebar:
        st.header("PDF Management")
        
        # PDF upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process PDFs", type="primary"):
                load_pdfs(uploaded_files)
        
        # PDF selection
        st.header("Select PDF")
        pdf_options = list(st.session_state.pdf_docs.keys())
        if pdf_options:
            selected_pdf = st.selectbox(
                "Choose a PDF to work with",
                options=pdf_options,
                index=0 if st.session_state.current_pdf is None else pdf_options.index(st.session_state.current_pdf)
            )
            
            if selected_pdf != st.session_state.current_pdf:
                update_preview(selected_pdf)
        
        # Quick actions
        if pdf_options:
            st.header("Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Summary"):
                    if st.session_state.current_pdf:
                        st.session_state.generated_content = generate_summary(st.session_state.current_pdf)
                        st.session_state.content_type = "summary"
            
            with col2:
                if st.button("Create Study Notes"):
                    if st.session_state.current_pdf:
                        st.session_state.generated_content = generate_notes(st.session_state.current_pdf)
                        st.session_state.content_type = "notes"
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Left column: PDF preview and query input
    with col1:
        st.header("PDF Preview")
        
        if st.session_state.current_pdf:
            st.markdown(f"**Current PDF**: {st.session_state.current_pdf}")
            st.text_area(
                "Preview",
                value=st.session_state.preview_text,
                height=250,
                disabled=True
            )
        else:
            st.info("Select a PDF from the sidebar to preview its contents")
        
        st.header("Ask Questions")
        
        query = st.text_area(
            "Enter your question or type 'Summarize this' for a summary",
            height=100
        )
        
        if st.button("Submit Query", type="primary"):
            if not query.strip():
                st.warning("Please enter a query")
            elif not st.session_state.current_pdf:
                st.warning("Please select a PDF first")
            else:
                if query.lower().strip() in ["summarize this", "summarize", "summary"]:
                    st.session_state.generated_content = generate_summary(st.session_state.current_pdf)
                    st.session_state.content_type = "summary"
                else:
                    st.session_state.generated_content = process_query(query, st.session_state.current_pdf)
                    st.session_state.content_type = "answer"
    
    # Right column: Generated content
    with col2:
        st.header("Generated Content")
        
        if "generated_content" in st.session_state and "content_type" in st.session_state:
            content_type = st.session_state.content_type
            content = st.session_state.generated_content
            
            content_title = {
                "summary": "📝 Summary",
                "notes": "📒 Study Notes",
                "answer": "🔍 Answer"
            }.get(content_type, "Generated Content")
            
            st.markdown(f"### {content_title}")
            st.markdown(content)
            
            # Download button
            file_extension = "md" if content_type in ["summary", "notes"] else "txt"
            filename = f"{st.session_state.current_pdf.split('.')[0]}_{content_type}.{file_extension}"
            
            st.download_button(
                label=f"Download {content_type.capitalize()}",
                data=content,
                file_name=filename,
                mime=f"text/{file_extension}"
            )
        else:
            st.info("Ask a question or generate content using the options provided")
    
    # Footer
    st.divider()
    st.markdown("""
    ### How to use this app:
    1. **Upload PDF(s)** using the sidebar and click "Process PDFs"
    2. **Select a PDF** from the dropdown menu
    3. **Ask questions** about the content or use quick actions to generate summaries and study notes
    4. **Download** the generated content for offline study
    """)

if __name__ == "__main__":
    main()