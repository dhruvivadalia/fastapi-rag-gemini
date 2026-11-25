import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "models/text-embedding-004"


def main() -> None:
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )

    doc_path = os.getenv("DOC_PATH") or "resume.pdf"
    pdf_file = Path(doc_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found at: {pdf_file.resolve()}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if not chunks:
        print(f"No text chunks extracted from '{doc_path}'.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_ids = [f"{pdf_file.name}-chunk-{i}" for i in range(len(chunks))]

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        ids=chunk_ids,
    )

    print(
        f"Stored {len(chunks)} chunks in Chroma collection "
        f"'{COLLECTION_NAME}' at '{CHROMA_DIR}'."
    )


if __name__ == "__main__":
    main()
