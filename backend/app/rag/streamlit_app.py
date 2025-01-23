import os
import sys
import logging
import nltk
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv
from chromadb import PersistentClient  # Updated import for Chroma client

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag.chunking_strategies import chunk_by_recursive_split

load_dotenv()

# Enhanced prompt template that explicitly asks for citations
GENERATOR_TEMPLATE = """
Use the following pieces of context to answer the question. 
Always include specific citations from the source documents to support your answer.
For each key point in your answer, reference the relevant document section.

Context: {context}

Question: {question}

Please provide a detailed answer with citations from the source documents.
"""

@dataclass
class CitedAnswer:
    """Data class to store the answer and its supporting citations."""
    answer: str
    citations: List[Dict[str, str]]
    confidence_score: float

def format_citations(source_documents: List[Document]) -> List[Dict[str, str]]:
    """
    Formats source documents into structured citations.
    
    Args:
        source_documents: List of source documents from the retriever
        
    Returns:
        List of citation dictionaries containing document metadata
    """
    citations = []
    for idx, doc in enumerate(source_documents):
        citation = {
            'text': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page_number', 'N/A')
        }
        citations.append(citation)
    return citations

class ChromaStore:
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Initialize Chroma client with the new configuration
        self.client = PersistentClient(path=data_path)

    def load_documents_from_dir(self) -> List[Document]:
        """
        Load documents from the specified directory.
        
        Returns:
            List of Document objects
        """
        # Implement your document loading logic here
        # Example: Load all .txt files from the directory
        documents = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append(Document(page_content=content, metadata={"source": file}))
        return documents

    def initialize_vectorstore(self, chunks: List[Document]):
        """
        Initialize the Chroma vector store with the provided chunks.
        
        Args:
            chunks: List of Document objects to add to the vector store
            
        Returns:
            Chroma collection or None if initialization fails
        """
        try:
            collection = self.client.get_or_create_collection(name="contracts")
            # Add documents to the collection
            for chunk in chunks:
                collection.add(
                    documents=[chunk.page_content],
                    metadatas=[chunk.metadata],
                    ids=[chunk.metadata.get("source", "unknown")]
                )
            return collection
        except Exception as e:
            logging.error(f"Failed to initialize vectorstore: {e}")
            return None

def initialize_rag_pipeline() -> Optional[RetrievalQA]:
    """
    Initializes an enhanced RAG pipeline with citation tracking capabilities.
    
    The pipeline now includes:
    - Document loading and chunking
    - Vector store initialization
    - Custom retrieval configuration
    - Citation-aware prompt template
    
    Returns:
        RetrievalQA: Enhanced pipeline that tracks source documents
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Download required NLTK resources
        nltk_resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng']
        for resource in nltk_resources:
            nltk.download(resource, quiet=True)

        # Initialize document store and load documents
        chroma_store = ChromaStore(
            data_path="/Users/venkata/ai-apps/contract-qa-high-precision-rag/backend/data/contracts"
        )
        documents = chroma_store.load_documents_from_dir()

        if not documents:
            logger.error("No documents loaded. Please check the data directory.")
            return None

        # Create semantic chunks for better retrieval
        chunks = chunk_by_recursive_split(documents, chunk_size=400)
        
        # Initialize vector store with enhanced metadata
        vectorstore = chroma_store.initialize_vectorstore(chunks)
        if vectorstore is None:
            logger.error("Failed to initialize vectorstore.")
            return None

        # Configure retriever with citation awareness
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5  # Number of documents to retrieve
            }
        )

        # Initialize language model with controlled temperature
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Create citation-aware prompt
        prompt = PromptTemplate(
            template=GENERATOR_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Configure QA chain with citation tracking
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa

    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def query_rag_pipeline(qa: RetrievalQA, query: str) -> CitedAnswer:
    """
    Queries the RAG pipeline and returns a cited answer.
    
    Args:
        qa: The initialized RAG pipeline
        query: The user's question
        
    Returns:
        CitedAnswer: Contains the answer, supporting citations, and confidence score
    """
    try:
        if qa is None:
            return CitedAnswer(
                answer="RAG pipeline is not initialized.",
                citations=[],
                confidence_score=0.0
            )

        # Get response with source documents
        response = qa.invoke({"query": query})
        answer = response["result"]
        source_docs = response["source_documents"]

        # Format citations from source documents
        citations = format_citations(source_docs)

        # Calculate a simple confidence score based on citation count
        confidence_score = min(len(citations) / 5.0, 1.0)

        return CitedAnswer(
            answer=answer,
            citations=citations,
            confidence_score=confidence_score
        )

    except Exception as e:
        logging.error(f"Failed to query RAG pipeline: {e}")
        return CitedAnswer(
            answer="An error occurred while processing your query.",
            citations=[],
            confidence_score=0.0
        )

def display_cited_answer(cited_answer: CitedAnswer):
    """
    Displays the answer with formatted citations in the Streamlit app.
    
    Args:
        cited_answer: CitedAnswer object containing the answer and citations
    """
    st.subheader("Answer:")
    st.write(cited_answer.answer)
    st.write(f"**Confidence Score:** {cited_answer.confidence_score:.1%}")
    
    if cited_answer.citations:
        st.subheader("Supporting Citations:")
        for idx, citation in enumerate(cited_answer.citations, 1):
            st.write(f"**[{idx}] Source:** {citation['source']}, **Page:** {citation['page']}")
            st.write(f"**Relevant text:** {citation['text']}")

def main():
    st.title("Contract Analysis System")
    st.write("Type your query about the contract below.")

    # Initialize the pipeline
    qa = initialize_rag_pipeline()

    # Input query
    query = st.text_input("Enter your query about the contract:")

    if st.button("Submit"):
        if not query:
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Processing your query..."):
                cited_answer = query_rag_pipeline(qa, query)
                display_cited_answer(cited_answer)

    # Sidebar for help and examples
    st.sidebar.title("Help & Examples")
    st.sidebar.write("**Available Commands:**")
    st.sidebar.write("- Type your query in the text box and click 'Submit'.")
    st.sidebar.write("- Use the examples below for inspiration.")
    
    st.sidebar.write("**Example Queries:**")
    st.sidebar.write("- What are the payment terms in the contract?")
    st.sidebar.write("- What are the obligations of the Advisor?")
    st.sidebar.write("- What is the termination policy?")

if __name__ == "__main__":
    main()