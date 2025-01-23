import os
import sys
import logging
import nltk

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag.chroma_store import ChromaStore
from rag.chunking_strategies import chunk_by_recursive_split
from misc import Settings

load_dotenv()

GENERATOR_TEMPLATE = Settings.GENERATOR_TEMPLATE

def initialize_rag_pipeline():
    """
    Initializes the Retrieval-Augmented Generation (RAG) pipeline by loading documents,
    chunking them, initializing a vector store, and setting up the retrieval and question-answering components.

    Returns:
        RetrievalQA: The initialized RAG pipeline ready for querying.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load the documents from the data directory.
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        chroma_store = ChromaStore(data_path="/Users/venkata/ai-apps/contract-qa-high-precision-rag/backend/data/contracts")
        documents = chroma_store.load_documents_from_dir()

        if not documents:
            logger.error("No documents loaded. Please check the data directory.")
            return None

        chunks = chunk_by_recursive_split(documents, chunk_size=400)

        vectorstore = chroma_store.initialize_vectorstore(chunks)
        if vectorstore is None:
            logger.error("Failed to initialize vectorstore.")
            return None

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
            }
        )

        PROMPT = PromptTemplate(
            template=GENERATOR_TEMPLATE, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT, "verbose": False}

        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            chain_type_kwargs=chain_type_kwargs,
        )
        return qa
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def query_rag_pipeline(qa, query):
    """
    Queries the RAG pipeline with a given question and returns the answer.

    Parameters:
        qa (RetrievalQA): The initialized RAG pipeline.
        query (str): The query or question to be answered.

    Returns:
        str: The answer to the query, or an error message if the query fails.
    """
    try:
        if qa is None:
            return "RAG pipeline is not initialized."
        response = qa.invoke({"query": query})
        answer = response["result"]
        return answer
    except Exception as e:
        logging.error(f"Failed to query RAG pipeline: {e}")
        return "An error occurred while processing your query."

# Example usage
if __name__ == "__main__":
    qa = initialize_rag_pipeline()
    if qa:
        query = "Can the Advisor charge for meal time?"
        answer = query_rag_pipeline(qa, query)
        print("Answer:", answer)
    else:
        print("Failed to initialize RAG pipeline.")


if __name__ == "__main__":
    print("\nContract Analysis System")
    print("=" * 80)
    print("Type 'help' or '?' for available commands and example queries.")
    print("=" * 80)
    
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
            response = query_rag_pipeline(qa, query)
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")