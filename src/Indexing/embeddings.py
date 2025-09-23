from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from Indexing.text_splitting import split

# Directory where the vector database will be stored
CHROMA_DIR = Path("data/chroma_db")
CHROMA_DIR.mkdir(exist_ok=True, parents=True)

# Use a small, local embedding model 
def create_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  
    )
    return embeddings


def store_in_vectorstore(documents):
    embeddings = create_embeddings_model()

    # Create (or load existing) Chroma vector store
    vectorstore = Chroma(
        collection_name="rag_app_docs",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    # Add documents (LangChain Document objects) into Chroma
    vectorstore.add_documents(documents)

    print(f"✅ Stored {len(documents)} documents in Chroma DB at {CHROMA_DIR}")


def build_vector_db():
    # Step 1: Creating Chunks documents for embeddings
    docs = split()

    # Step 2: Store in Chroma
    store_in_vectorstore(docs)


if __name__ == "__main__":
    build_vector_db()
