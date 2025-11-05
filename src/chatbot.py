"""
RAG Chatbot Application with Sentence-BERT, FAISS, and Groq
===========================================================

A modular chatbot system that implements Retrieval-Augmented Generation (RAG)
using Sentence-BERT for embeddings, FAISS for vector search, and Groq for LLM inference.

Installation: pip install sentence-transformers faiss-cpu groq langchain langchain-core numpy
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

# Core dependencies
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import groq

# LangChain components
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    from langchain.schema import Document
    from langchain.schema.retriever import BaseRetriever
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqLLM(LLM):
    """Custom LangChain LLM wrapper for Groq API using Llama 4 Scout."""
    
    api_key: str
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    client: Optional[Any] = None
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", **kwargs):
        # Initialize fields before calling super()._init_
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            client=None,
            **kwargs
        )
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Groq client."""
        try:
            self.client = groq.Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
        
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    @property 
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "api_key_set": bool(self.api_key)
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Groq API."""
        if self.client is None:
            return "Error: Groq client not initialized. Please check your API key."
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"Error: Unable to generate response. {str(e)}"

class DocumentLoader:
    """Handles loading and processing of different document types."""
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_json_file(file_path: str) -> str:
        """Load and stringify JSON file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to readable text format
                if isinstance(data, dict):
                    return DocumentLoader._dict_to_text(data)
                elif isinstance(data, list):
                    return "\n\n".join([DocumentLoader._dict_to_text(item) if isinstance(item, dict) else str(item) for item in data])
                else:
                    return str(data)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_markdown_file(file_path: str) -> str:
        """Load content from a Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading Markdown file {file_path}: {e}")
            return ""
    
    @staticmethod
    def _dict_to_text(data: Dict) -> str:
        """Convert dictionary to readable text format."""
        text_parts = []
        for key, value in data.items():
            if isinstance(value, dict):
                text_parts.append(f"{key}:\n{DocumentLoader._dict_to_text(value)}")
            elif isinstance(value, list):
                text_parts.append(f"{key}: {', '.join(map(str, value))}")
            else:
                text_parts.append(f"{key}: {value}")
        return "\n".join(text_parts)
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents of different types."""
        documents = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            content = ""
            file_extension = path.suffix.lower()
            
            if file_extension == '.txt':
                content = self.load_text_file(file_path)
            elif file_extension == '.json':
                content = self.load_json_file(file_path)
            elif file_extension in ['.md', '.markdown']:
                content = self.load_markdown_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                continue
            
            if content:
                # Create LangChain document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "file_type": file_extension,
                        "file_name": path.name
                    }
                )
                documents.append(doc)
        
        return documents

class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def build_index(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
        """Build FAISS index from documents."""
        logger.info(f"Building FAISS index from {len(documents)} documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        self.documents = chunked_docs
        logger.info(f"Created {len(chunked_docs)} document chunks")
        
        # Generate embeddings
        texts = [doc.page_content for doc in chunked_docs]
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.embeddings = embeddings
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                doc = self.documents[idx]
                results.append((doc, float(score)))
        
        return results
    
    def save_index(self, index_path: str, documents_path: str):
        """Save FAISS index and documents to disk."""
        if self.index is None:
            raise ValueError("No index to save")
            
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Index saved to {index_path}, documents to {documents_path}")
    
    def load_index(self, index_path: str, documents_path: str):
        """Load FAISS index and documents from disk."""
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Index loaded from {index_path}, documents from {documents_path}")

class CustomRetriever(BaseRetriever):
    """Custom retriever that wraps FAISSVectorStore."""
    
    vector_store: FAISSVectorStore
    k: int = 4
    
    def __init__(self, vector_store: FAISSVectorStore, k: int = 4, **kwargs):
        super().__init__(
            vector_store=vector_store,
            k=k,
            **kwargs
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        results = self.vector_store.similarity_search(query, k=self.k)
        return [doc for doc, score in results]

class RAGChatbot:
    """Main RAG Chatbot class that orchestrates all components."""
    
    def __init__(self, groq_api_key: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.groq_api_key = groq_api_key
        self.embedding_model_name = embedding_model_name
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.vector_store = FAISSVectorStore(embedding_model_name)
        self.llm = GroqLLM(groq_api_key)
        
        # LangChain components
        self.retriever = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        
    def load_and_index_documents(self, file_paths: List[str], chunk_size: int = 1000):
        """Load documents and build search index."""
        logger.info("Loading documents...")
        documents = self.document_loader.load_documents(file_paths)
        
        if not documents:
            raise ValueError("No documents loaded successfully")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build index
        self.vector_store.build_index(documents, chunk_size=chunk_size)
        
        # Create retriever
        self.retriever = CustomRetriever(self.vector_store, k=4)
        
        # Create QA chain
        self._create_qa_chain()
        
    def _create_qa_chain(self):
        """Create the conversational retrieval chain."""
        # Custom prompt template
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer based on the context, just say that you don't know.
        Be conversational and helpful while staying grounded in the provided information.

        Context: {context}

        Question: {question}
        
        Answer:"""
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        if self.qa_chain is None:
            raise ValueError("No documents indexed. Call load_and_index_documents() first.")
        
        try:
            # Get response from chain
            response = self.qa_chain({"question": question})
            
            # Format response
            result = {
                "answer": response["answer"],
                "sources": []
            }
            
            # Add source information
            for doc in response.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                result["sources"].append(source_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": []
            }
    
    def save_index(self, index_dir: str = "rag_index"):
        """Save the current index for later use."""
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, "faiss.index")
        docs_path = os.path.join(index_dir, "documents.pkl")
        self.vector_store.save_index(index_path, docs_path)
    
    def load_index(self, index_dir: str = "rag_index"):
        """Load a previously saved index."""
        index_path = os.path.join(index_dir, "faiss.index")
        docs_path = os.path.join(index_dir, "documents.pkl")
        self.vector_store.load_index(index_path, docs_path)
        self.retriever = CustomRetriever(self.vector_store, k=4)
        self._create_qa_chain()

def create_sample_documents():
    """Create sample documents for demonstration."""
    
    # Sample text document
    sample_text = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is a broad field of computer science that aims to create 
    systems capable of performing tasks that typically require human intelligence. These 
    tasks include visual perception, speech recognition, decision-making, and language translation.
    
    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
    that can learn and improve from experience without being explicitly programmed. ML 
    algorithms build mathematical models based on training data to make predictions or 
    decisions without being specifically programmed to perform the task.
    
    Deep Learning is a subset of machine learning that uses artificial neural networks 
    with multiple layers (hence "deep") to model and understand complex patterns in data.
    """
    
    # Sample JSON document
    sample_json = {
        "company": "TechCorp AI",
        "products": [
            {
                "name": "SmartBot",
                "type": "Chatbot",
                "features": ["Natural Language Processing", "Multi-language Support", "24/7 Availability"],
                "description": "An advanced AI chatbot for customer service automation"
            },
            {
                "name": "VisionAI",
                "type": "Computer Vision",
                "features": ["Object Detection", "Facial Recognition", "Real-time Processing"],
                "description": "AI-powered image and video analysis platform"
            }
        ],
        "about": "TechCorp AI specializes in developing cutting-edge artificial intelligence solutions for businesses across various industries."
    }
    
    # Sample Markdown document
    sample_markdown = """
    # RAG Systems Guide
    
    ## What is RAG?
    
    Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more accurate, contextual responses.
    
    ## Key Components
    
    ### 1. Document Retrieval
    - Vector embeddings for semantic search
    - Similarity matching algorithms
    - Efficient indexing systems like FAISS
    
    ### 2. Text Generation
    - Large Language Models (LLMs)
    - Context-aware response generation
    - Prompt engineering techniques
    
    ## Benefits
    
    - *Accuracy*: Responses grounded in factual information
    - *Scalability*: Can work with large document collections
    - *Flexibility*: Easy to update knowledge base
    - *Transparency*: Provides source attribution
    """
    
    # Write sample files
    os.makedirs("sample_docs", exist_ok=True)
    
    with open("sample_docs/ai_basics.txt", "w") as f:
        f.write(sample_text)
    
    with open("sample_docs/company_info.json", "w") as f:
        json.dump(sample_json, f, indent=2)
    
    with open("sample_docs/rag_guide.md", "w") as f:
        f.write(sample_markdown)
    
    logger.info("Sample documents created in 'sample_docs' directory")

def main():
    """RAG Chatbot using scout_results.txt as input."""

    # Set up Groq API key
    GROQ_API_KEY = "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"  # <-- replace with your key

    if GROQ_API_KEY == "your-groq-api-key-here":
        print("Please set your Groq API key in the GROQ_API_KEY variable")
        return

    try:
        # Initialize chatbot
        print("Initializing RAG Chatbot...")
        chatbot = RAGChatbot(groq_api_key=GROQ_API_KEY)

        # Load and index scout_results.txt
        document_files = ["scout_results.txt"]

        print("Loading and indexing documents from scout_results.txt...")
        chatbot.load_and_index_documents(document_files)

        # Save index for later use
        print("Saving index...")
        chatbot.save_index()

        # Interactive mode
        print("\n" + "=" * 50)
        print("RAG CHATBOT (scout_results.txt)")
        print("=" * 50)
        print("Type 'quit' to exit.\n")

        while True:
            user_question = input("\nYour question: ").strip()
            if user_question.lower() in ["quit", "exit", "q"]:
                break

            if user_question:
                response = chatbot.ask(user_question)
                print(f"\nAnswer: {response['answer']}")

                if response["sources"]:
                    print(f"\nSources ({len(response['sources'])} found):")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"{i}. {source['metadata']['file_name']}: {source['content']}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()


