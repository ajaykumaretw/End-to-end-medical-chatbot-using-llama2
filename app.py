from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import List

from src.prompt import *
import os
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"

embedding_model = download_hugging_face_embeddings()

# Initialize Pinecone with the API key
if api_key:
    pc = Pinecone()
    print(f"Pinecone Initialized: {pc}")
else:
    print("Error: API key not found.")

pinecone_index = pc.Index(index_name)  # Renamed 'index' to 'pinecone_index'

# Retrieve index info to get the host
index_info = pc.describe_index(index_name)
print(index_info)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

def query_pinecone(query_text, top_k=5):
    # Generate the embedding for the query text
    query_embedding = embedding_model.embed_query(query_text)
    # Perform the query using the embedding
    query_result = pinecone_index.query(  # Updated to use 'pinecone_index'
        vector=query_embedding,
        top_k=top_k,
        namespace='ns_medical_chatbot',
        include_metadata=True,
    )
    return query_result

class CustomPineconeRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = query_pinecone(query)
        docs = []
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.pop('text', '')
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

custom_retriever = CustomPineconeRetriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
