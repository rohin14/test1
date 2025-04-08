# rag_pipeline.py
# This module handles the RAG pipeline for query processing and generation

import os
from typing import List, Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


class RAGPipeline:
    """Handles the RAG pipeline for query processing and generation"""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama3-70b-8192"):
        """
        Initialize the RAG pipeline
        
        Args:
            groq_api_key: API key for the GROQ API
            model_name: Name of the GROQ model to use
        """
        # Set the API key
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize the LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.2
        )
    
    def create_qa_chain(self, retriever):
        """
        Create a question-answering chain
        
        Args:
            retriever: Retriever object from the vector store
            
        Returns:
            QA chain that can be invoked with a question
        """
        # Define the prompt template
        qa_prompt = ChatPromptTemplate.from_template(
            """You are a helpful academic assistant that answers questions based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            - Answer the question based only on the provided context
            - If the context doesn't contain the answer, say "I don't have enough information to answer this question."
            - Be concise and accurate
            - Use specific examples from the context when appropriate
            - Cite the page number when referencing specific information (e.g., "According to page 3...")
            
            Answer:"""
        )
        
        # Create the chain
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return qa_chain
    
    def create_summary_chain(self, retriever):
        """
        Create a summarization chain
        
        Args:
            retriever: Retriever object from the vector store
            
        Returns:
            Summary chain that can be invoked with a document
        """
        # Define the prompt template
        summary_prompt = ChatPromptTemplate.from_template(
            """You are an expert at summarizing academic content.
            
            Content to summarize:
            {context}
            
            Instructions:
            - Provide a comprehensive summary of the provided content
            - Identify and include the key points, main ideas, and essential information
            - Organize the summary in a logical structure
            - Keep the summary informative yet concise
            - Maintain the academic tone of the original content
            
            Summary:"""
        )
        
        # Create the chain
        summary_chain = (
            {"context": retriever}
            | summary_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return summary_chain
    
    def create_notes_chain(self, retriever):
        """
        Create a notes generation chain
        
        Args:
            retriever: Retriever object from the vector store
            
        Returns:
            Notes chain that can be invoked with a document
        """
        # Define the prompt template
        notes_prompt = ChatPromptTemplate.from_template(
            """You are an expert at creating study notes for students.
            
            Content to convert to notes:
            {context}
            
            Instructions:
            - Create comprehensive study notes from the provided content
            - Structure the notes with clear headings, subheadings, and bullet points
            - Include all important concepts, definitions, theories, and examples
            - Organize information hierarchically with main points and supporting details
            - Format in Markdown with proper headers (##, ###), bullet points, and emphasis
            - Include any relevant formulas, diagrams descriptions, or key quotations
            - Add a "Key Takeaways" section at the end
            
            Study Notes:"""
        )
        
        # Create the chain
        notes_chain = (
            {"context": retriever}
            | notes_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return notes_chain