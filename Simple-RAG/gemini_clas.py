import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from dotenv import load_dotenv
from typing import List
import chromadb

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
path = ""


class GeminiEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("api key ulaşılamadı")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(
            model=model, content=input, task_type="retrieval_document", title=title
        )["embedding"]


def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction()
    )

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    return db, name


def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(
        name=name, embedding_function=GeminiEmbeddingFunction()
    )

    return db
