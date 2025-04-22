import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from groq import Groq
import pdfplumber
import docx
from bs4 import BeautifulSoup
import glob

def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    text = ""

    if ext == "txt":
        with open(file_path, 'r', encoding="utf-8") as f:
            text = f.read()

    elif ext == "pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif ext == "docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif ext in ["html", "htm"]:
        with open(file_path, 'r', encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()

    elif ext == "md":
        with open(file_path, 'r', encoding="utf-8") as f:
            text = f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return text.strip()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# NEW WAY: Initialize ChromaDB client
chroma_client = PersistentClient(path="./chroma_storage")

# Initialize Sentence Transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Folder where your files are stored
folder_path = "./documents"

# Find all text-based files
file_paths = glob.glob(folder_path + "/*")

documents = []
doc_ids = []

for i, file_path in enumerate(file_paths):
    try:
        text = extract_text_from_file(file_path)
        if text:  # only add non-empty docs
            documents.append(text)
            doc_ids.append(f"doc_{i}")
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")

# Now embed and add them
doc_embeddings = embedder.encode(documents).tolist()

collection = chroma_client.get_or_create_collection(name="rag_collection")

collection.add(
    documents=documents,
    embeddings=doc_embeddings,
    ids=doc_ids
)

# 1. User Query
user_query = "Who created Gridgenius?"

# 2. Embed the query
query_embedding = embedder.encode([user_query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,  # get top 3 matches
    include=["documents"]
)

retrieved_docs = results['documents'][0]  # List of top 3 docs
combined_context = "\n\n".join(retrieved_docs)  # Merge them nicely

retrieved_doc = results['documents'][0][0]
print(f"Document Retrieved")

# 4. Send to Groq for Answering
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant", 
    messages=[
        {"role": "system", "content": f"You are an expert assistant. Use the following context to answer the question:\n\n{combined_context}"},
        {"role": "user", "content": user_query}
    ] 
)

print("\nGroq's Answer:")
print(response.choices[0].message.content)
