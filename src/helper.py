from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import json
# Extract data from PDF file
def load_pdf(data):
 loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
 documents=loader.load()
 return documents

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def flatten_metadata(metadata):
    """Flatten metadata to simple key-value pairs."""
    flattened = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, list) and all(isinstance(i, str) for i in value):
            flattened[key] = value
        else:
            # Convert complex structures to JSON strings
            flattened[key] = json.dumps(value)
    return flattened

# Helper function to split vectors into smaller batches
def batch_vectors(vectors, batch_size):
    for i in range(0, len(vectors), batch_size):
        yield vectors[i:i + batch_size]
        
# Function to check if the index exists
def index_exists(client, index_name):
    try:
        # Try to describe the index; if it exists, this will return index info
        client.describe_index(index_name)
        return True
    except Exception as e:
        # If the index doesn't exist, describe_index will raise an error
        return False