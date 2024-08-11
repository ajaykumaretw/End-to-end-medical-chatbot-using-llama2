from src.helper import load_pdf,text_split,download_hugging_face_embeddings,flatten_metadata,batch_vectors,index_exists
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
import pinecone
from dotenv import load_dotenv
import os
import uuid
import math
import json

# Load environment variables from a .env file
load_dotenv()


# Retrieve the Pinecone API key from the environment variables
api_key = os.getenv("PINECONE_API_KEY")

index_name = "medical-chatbot"
namespace="ns_medical_chatbot"
dimension = 384  # Dimension of the vectors you're working with
metric = "cosine"  # Similarity metric

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

embedding_model = download_hugging_face_embeddings()

# Generate embeddings for each text chunk
embeddings = [embedding_model.embed_query(t.page_content) for t in text_chunks]

# Initialize Pinecone with the API key
if api_key:
    pc = Pinecone()
    print(f"Pinecone Initialized: {pc}")
else:
    print("Error: API key not found.")
    
# Check if the index already exists
if index_exists(pc, index_name):
    print(f"Index '{index_name}' already exists.")
else:
    # Create the index using the API key stored in the client 'pc'
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        )
    )
    print(f"Index '{index_name}' created successfully.")

# Access the index
index = pc.Index(index_name)

# Retrieve and print index information to get details such as the host
index_info = pc.describe_index(index_name)
print(index_info)


# Define batch size (adjust as needed)
batch_size = 1000  # Set a batch size that fits within Pinecone's limits

# Prepare the batch of vectors to upsert
vectors = [(str(uuid.uuid4()), list(embedding), {
    "text": chunk.page_content,
    **flatten_metadata(chunk.metadata),
}) for embedding, chunk in zip(embeddings, text_chunks)]

# Upsert vectors in batches
for batch in batch_vectors(vectors, batch_size):
    index.upsert(vectors=batch,namespace=namespace)
    print(f"Uploaded {len(batch)} vectors to Pinecone index '{index}'")

print(f"Uploaded a total of {len(vectors)} vectors to Pinecone index '{index}'")
