import os
import sys
import logging
import nltk
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag.chroma_store import ChromaStore
from rag.chunking_strategies import chunk_by_recursive_split
from misc import Settings

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
            data_path="/Users/venkata/ai-apps/legal-query/backend/data/contracts"
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

def print_cited_answer(cited_answer: CitedAnswer):
    """
    Prints the answer with formatted citations.
    
    Args:
        cited_answer: CitedAnswer object containing the answer and citations
    """
    print("\nAnswer:")
    print("=" * 80)
    print(cited_answer.answer)
    print("\nConfidence Score: {:.1%}".format(cited_answer.confidence_score))
    
    if cited_answer.citations:
        print("\nSupporting Citations:")
        print("-" * 80)
        for idx, citation in enumerate(cited_answer.citations, 1):
            print(f"\n[{idx}] Source: {citation['source']}, Page: {citation['page']}")
            print(f"Relevant text: {citation['text']}")

def print_help():
    """Prints help information and example queries."""
    print("\nAvailable Commands:")
    print("- 'help' or '?': Show this help message")
    print("- 'quit', 'exit', or 'q': Exit the program")
    
    print("\nExample Queries:")
    print("- What are the payment terms in the contract?")
    print("- What are the obligations of the Advisor?")
    print("- What is the termination policy?")

if __name__ == "__main__":
    print("\nContract Analysis System")
    print("=" * 80)
    print("Type 'help' or '?' for available commands and example queries.")
    print("=" * 80)
    
    # Initialize the pipeline
    qa = initialize_rag_pipeline()
    
    # Main input loop
    while True:
        try:
            query = input("\nEnter your query about the contract: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting the program...")
                break
            elif query.lower() in ['help', '?']:
                print_help()
                continue
            elif not query:
                print("\nPlease enter a valid query.")
                continue
                
            print("\nProcessing your query...\n")
            cited_answer = query_rag_pipeline(qa, query)
            print_cited_answer(cited_answer)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")