from dotenv import load_dotenv
from mem0 import Memory
import os
from openai import OpenAI
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HOST_QDRANT_URL = os.getenv("HOST_QDRANT_URL")

client = OpenAI()

base_config = {
    "version": "v1.1",

    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small"
        }
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "gpt-turbo-3.5"}},
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": HOST_QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "https": True
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://neo4j:7687",
            "username": "neo4j",
            "password": "reform-william-center-vibrate-press-5829"
        }
    },
}

# mem_client = Memory.from_config(base_config)
