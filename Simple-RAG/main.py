import os
from pypdf import PdfReader
import re
from gemini_clas import (
    create_chroma_db,
    load_chroma_collection,
    GeminiEmbeddingFunction,
)
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
import google.generativeai as genai


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text


pdf_text = load_pdf(file_path="")


def split_text_with_overlap(text, chunk_size=500, overlap=50):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + " "
        else:
            while len(para) > 0:
                chunk = (
                    current_chunk + para[: chunk_size - len(current_chunk)]
                ).strip()
                chunks.append(chunk)
                para = para[chunk_size - len(current_chunk) - overlap :]
                current_chunk = para[:overlap]

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


chunked_text = split_text_with_overlap(pdf_text, 500, 50)

db, name = create_chroma_db(
    documents=chunked_text,
    path="",
    name="rag_experiment4",
)


def get_relevant_passage(query, db, n_results):
    passage = db.query(query_texts=[query], n_results=n_results)["documents"][0]
    return passage


def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (
        """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

  ANSWER:
  """
    ).format(query=query, relevant_passage=escaped)

    return prompt


def generate_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable"
        )
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    answer = model.generate_content(prompt)
    return answer.text


db = load_chroma_collection(
    path="",
    name="rag_experiment4",
)
query = ""

relevant_text = get_relevant_passage(query, db, n_results=3)

prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))

answer = generate_answer(prompt)

print(answer)
