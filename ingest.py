import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# IMPORTANT: Set up your Google API Key as an environment variable
# before running this script.
# On macOS/Linux: export GOOGLE_API_KEY="YOUR_API_KEY"
# On Windows: set GOOGLE_API_KEY="YOUR_API_KEY"
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDgBe4ibRkgIyAyTFkZuV8uiPMbIrKwRfU" # Fallback for convenience, but env var is better.

PDF_PATH = "dsm.pdf"
CHROMA_DB_PATH = "./chroma_db" # Path to store the vector database

def main():
    print("Starting data ingestion process...")

    # 1. Load the PDF document
    print(f"Loading PDF from: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    if not documents:
        print("Error: Could not load any documents from the PDF.")
        return
    print(f"Successfully loaded {len(documents)} pages from the PDF.")

    # 2. Split the document into smaller chunks for processing
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split document into {len(texts)} text chunks.")

    # 3. Create Google Generative AI embeddings
    print("Creating embeddings with Google Generative AI...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create a Chroma vector store and ingest the documents
    # This will create a local folder `chroma_db` to store the embeddings.
    print(f"Creating and persisting vector store at: {CHROMA_DB_PATH}")
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print("Vector store created successfully.")
    
    # Verify the count
    count = vector_store._collection.count()
    print(f"Total documents ingested into ChromaDB: {count}")

    print("\nData ingestion complete!")
    print("You can now run the `app.py` server to start the chatbot.")

if __name__ == "__main__":
    main()
