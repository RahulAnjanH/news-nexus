import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #imports
from langchain_ollama  import OllamaEmbeddings 
from langchain_chroma import Chroma

DATA_PATH=r"D:\rahul_learning\python-project\news-nexus\data\raw_pdfs"
DB_PATH=r"D:\rahul_learning\python-project\news-nexus\data\chroma_db"
def ingest_documents():
    #Load doucments
    print(f"Loading PDF's from {DATA_PATH}")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages....")
    #Split Text
    text_spitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=50,                       # Data is loaded
        length_function = len,
        is_separator_regex=False
    )
    chunks = text_spitter.split_documents(raw_documents)
    print(f"Spilt into {len(chunks)} chunks")
    #Initilize Embeddings
    embedding_model = OllamaEmbeddings(models="nomic-embed-text")   #Embeddings
    print("Initilizing Vector Store (this may take a few minutes for large PDFs)...")
    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    BATCH_SIZE = 100
    total_chunks = len(chunks)
    for i in range(0,total_chunks,BATCH_SIZE):
        batch = chunks[i:i:i+BATCH_SIZE]  #Batch processing
        print(f"Processing Batch {BATCH_SIZE} of {total_chunks}")
        vector_db.add_documents(batch)
    print("Vector Store created successfully")
    return len(raw_documents),len(chunks)

if __name__ == "__main__":
    os.makedirs(DATA_PATH,exist_ok=True) #main

    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"No PDF's found in {DATA_PATH} Please add files to enable RAG")
    else:                                       
        ingest_documents()    #calling documents