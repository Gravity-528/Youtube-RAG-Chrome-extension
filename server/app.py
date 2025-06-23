from fastapi import FastAPI,Query
from .ingestion import graph_builder
app=FastAPI()

@app.get('/')
def func():
    print("hello how are you")


@app.post('/chat')
def chat(
    query:str= Query(...,description="chat message")
):
    pass

@app.post("/ingest")
def ingest_documents(
    url:str = Query(..., description="URL to ingest documents from"),
    type: str = Query(..., description="Type of ingestion (e.g., 'doc'or 'transcript', etc.)"),
    video_id: str = Query(None, description="Video ID for transcript ingestion")
):
    
    _state = {
        "url": url,
        "type": type,
        "video_id": video_id,
        "list_to_scrap": []
    }
    graph_builder(_state)
    return {"message": "Documents ingested successfully"}

@app.get("/query_rag_answer")
def query_rag_answer(
    query: str = Query(..., description="Query to ask the RAG model")
):
    # Placeholder for the actual query logic
    return {"message": "Query received successfully", "query": query}