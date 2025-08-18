from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_file, filter_to_minimal_docs, text_split,
    download_hugging_face_embeddings, load_csv_files, clean_csv
)

#  LOAD ENV 
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#  EMBEDDINGS 
embeddings = download_hugging_face_embeddings()

#  PINECONE INIT 
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

#  INDEX PDFs 
extracted_pdf_data = load_pdf_file(data="data/")
filter_pdf_data = filter_to_minimal_docs(extracted_pdf_data)
text_pdf_chunks = text_split(filter_pdf_data)

docsearch = PineconeVectorStore.from_documents(
    documents=text_pdf_chunks,
    embedding=embeddings,
    index_name=index_name
)

# INDEX CSV
df = load_csv_files("data/synthetic_doctor_database.csv")
csv_docs = clean_csv(df)
docsearch.add_documents(csv_docs)

print("PDFs and CSV data indexed successfully into Pinecone!")
