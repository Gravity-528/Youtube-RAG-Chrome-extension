import os
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
import re

load_dotenv()

_qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def safe_collection_name(email: str, link: str) -> str:
    email_part = email.replace('@', '_').replace('.', '_')
    
    link_part = re.sub(r'https?://', '', link)  
    link_part = re.sub(r'[^a-zA-Z0-9_]', '_', link_part)  

    return f"user_{email_part}_{link_part}"

# def get_vectorstore(email: str,link:str) -> Qdrant:
#     collection_name = safe_collection_name(email,link)

#     if not _qdrant_client.collection_exists(collection_name):
#         _qdrant_client.create_collection(
#             collection_name=collection_name,
#             vectors_config={
#                 "size": _embeddings.embed_query("test").__len__(),
#                 "distance": "Cosine"
#             }
#         )

#     return Qdrant(
#         client=_qdrant_client,
#         collection_name=collection_name,
#         embeddings=_embeddings
#     )

def get_vectorstore(email: str, link: str, force_recreate: bool = False) -> Qdrant:
    collection_name = safe_collection_name(email, link)

    if force_recreate and _qdrant_client.collection_exists(collection_name):
        print(f"[INFO] Deleting existing collection: {collection_name}")
        _qdrant_client.delete_collection(collection_name=collection_name)

    if not _qdrant_client.collection_exists(collection_name):
        _qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": len(_embeddings.embed_query("test")),
                "distance": "Cosine"
            }
        )

    return Qdrant(
        client=_qdrant_client,
        collection_name=collection_name,
        embeddings=_embeddings
    )