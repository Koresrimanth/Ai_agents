import os
from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_documents():
    data_dir = 'data'
    documents = []

    # Load PDF files
    for file in os.listdir(data_dir):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())

    # Load Excel files
    for file in os.listdir(data_dir):
        if file.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory='chroma_db')
    vectordb.persist()
    print("Documents ingested and stored in ChromaDB.")

if __name__ == "__main__":
    ingest_documents()
