from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import pandas as pd
from typing import List

#  PDF FUNCTIONS 
def load_pdf_file(data: str):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only source + page_content in metadata."""
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs

def text_split(minimal_docs: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return text_splitter.split_documents(minimal_docs)

#  EMBEDDINGS 
def download_hugging_face_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

#  CSV FUNCTIONS 
def load_csv_files(data: str):
    """Read CSV into pandas dataframe."""
    return pd.read_csv(data)

def clean_csv(df: pd.DataFrame) -> List[Document]:
    """Convert dataframe rows into LangChain Document objects."""
    df_documents = []
    for _, row in df.iterrows():
        content = f"{row['doctor_name']} works in the {row['section']} section and is available at {row['timing']}."
        metadata = {
            "doctor_name": row['doctor_name'],
            "section": row['section'],
            "timing": row['timing']
        }
        df_documents.append(Document(page_content=content, metadata=metadata))
    return df_documents
