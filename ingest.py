from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

PDF_PATH = "S26_JuhiPatil_Resume.pdf"
CHROMA_DIR = "chroma_db"

def ingest():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    print(f"Splitting {len(pages)} page(s)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    print(f"Embedding {len(chunks)} chunks and storing...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)

    print("Done. Database saved to chroma_db/")

if __name__ == "__main__":
    ingest()