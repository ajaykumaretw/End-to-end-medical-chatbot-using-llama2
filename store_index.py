from src.helper import load_pdf,text_split,download_hugging_face_embeddings,flatten_metadata,batch_vectors
from langchain.vectorstores import Pinecone
from pinecone import ServerlessSpec
import pinecone
from dotenv import load_dotenv
import os
import uuid
import math
import json

load_dotenv()

#PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV='gcp-starter'
# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
print(api_key)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

embedding_model = download_hugging_face_embeddings()

# Generate embeddings for each text chunk
embeddings = [embedding_model.embed_query(t.page_content) for t in text_chunks]


pc = Pinecone(api_key=api_key)

# Define index name
index_name = "medical-chatbot"
# Creates an index using the API key stored in the client 'pc'.
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws', 
        region='us-east-1'
    ) 
)

index = pc.Index(index_name)

# Retrieve index info to get the host
index_info = pc.describe_index(index_name)
print(index_info)

# Define batch size (adjust as needed)
batch_size = 1000  # Set a batch size that fits within Pinecone's limits

# Prepare the batch of vectors to upsert
vectors = [(str(uuid.uuid4()), list(embedding), {
    "text": chunk.page_content,
    **flatten_metadata(chunk.metadata),
}) for embedding, chunk in zip(embeddings, text_chunks)]
namespace="ns_medical_chatbot"
# Upsert vectors in batches
for batch in batch_vectors(vectors, batch_size):
    index.upsert(vectors=batch,namespace=namespace)
    print(f"Uploaded {len(batch)} vectors to Pinecone index '{index}'")

print(f"Uploaded a total of {len(vectors)} vectors to Pinecone index '{index}'")
