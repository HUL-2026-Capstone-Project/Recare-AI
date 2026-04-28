from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()
embedding = OpenAIEmbeddings()

docs_path = "docs"
all_chunks = []

for filename in os.listdir(docs_path):
    file_path = os.path.join(docs_path, filename)

    if filename.endswith(".txt"):
        try:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding="cp949") as f:
                text = f.read()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        ).split_text(text)
        all_chunks.extend(chunks)

    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for page in pages:
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            ).split_text(page.page_content)
            all_chunks.extend(chunks)

vectordb = FAISS.from_texts(all_chunks, embedding)
vectordb.save_local("vector_db")
print("✅ 벡터 DB 저장 완료!")
