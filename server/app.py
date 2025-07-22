import os,requests
os.environ.setdefault("INNGEST_DEV", "1")
from scrapy.utils.reactor import install_reactor
install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")
from fastapi import FastAPI,Query
from contextlib import asynccontextmanager
import logging
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from vectorstore import get_vectorstore
# from inngest import Event
import inngest
import inngest.fast_api
from ingestion import scrape_recursive, youtube_ingestion, doc_ingestion
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pydantic import BaseModel
from query_answer import build_langgraph
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request


# langfuse = Langfuse(
#   secret_key="sk-lf-cb7fc817-99a2-4857-a2ea-280070bfdf7e",
#   public_key="pk-lf-9c926bfc-d2f9-4af7-8cf9-63e000e62143",
#   host="https://cloud.langfuse.com"
# )


inngest_client = inngest.Inngest(
    app_id="rag_pipeline_worker",
    logger=logging.getLogger("uvicorn"),
)

langfuse_handler = CallbackHandler()
async def web_scrape_and_ingest(state, ctx: inngest.Context):
    ctx.logger.info(f"🔍 Web scraping for {state['url']}")
    items = scrape_recursive(state["url"],30)
    
    if not items:
        ctx.logger.warning("No documents found to ingest.")
        return
    
    ctx.logger.info(f"Ingesting {len(items)} documents for {state['email']} from {state['url']}")
    doc_ingestion(items, state["email"], state["url"])
    ctx.logger.info("Documents ingested successfully")


async def youtube_ingest(state, ctx: inngest.Context):
    ctx.logger.info(f"YouTube ingestion for video ID {state['video_id']}")
    try:
        youtube_ingestion(state["video_id"], state["email"])
    except TranscriptsDisabled:
        ctx.logger.warning("No transcripts available for this video")
        return
    ctx.logger.info(" YouTube ingestion completed successfully")


@inngest_client.create_function(
    fn_id="process_case",
    trigger=inngest.TriggerEvent(event="case/process"),
)
async def process_case(ctx: inngest.Context):
    state = ctx.event.data
    ctx.logger.info(f" Processing case: {state}")

    if state["type"] == "doc":
        await web_scrape_and_ingest(state, ctx)
    elif state["type"] == "transcript":
        await youtube_ingest(state, ctx)
    else:
        raise ValueError(f"Unsupported type {state['type']}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    yield 
    key = os.getenv("INNGEST_SIGNING_KEY")
    deploy_id = os.getenv("RENDER_GIT_COMMIT")
    base = os.getenv("RENDER_EXTERNAL_URL")
    if key and deploy_id and base:
        try:
            url = f"{base}/api/inngest?deployId={deploy_id}"
            resp = requests.put(url, headers={"Authorization": f"Bearer {key}"}, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            app.logger.error("Inngest sync failed:", exc_info=e)



app=FastAPI(lifespan=lifespan)

inngest.fast_api.serve(app, inngest_client, [process_case], serve_path="/api/inngest")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def func():
    print("hello how are you")

@app.post('/chat')
async def chat(
    query:str= Query(...,description="chat message")
):
    pass

class IngestRequest(BaseModel):
    email: str
    url: str
    type: str
    video_id: str | None = None
@app.post("/ingest")
async def ingest_documents(
    request: IngestRequest
):
    video_id = request.video_id
    email = request.email
    url = request.url
    type = request.type

    state = {
        "type": type,
        "video_id": video_id,
        "email": email,
        "url": url,
    }
    # await inngest_client.send(name="case/process", data=state)
    ids=await inngest_client.send(inngest.Event(name="case/process", data=state))
    return {"message": "Documents ingestion started...please wait for a few seconds"}

class QueryRAGRequest(BaseModel):
    query: str
    email: str
    url: str
@app.post("/query_rag_answer")
def query_rag_answer(
    request: QueryRAGRequest
):
    email = request.email
    url = request.url
    vectorstore= get_vectorstore(email=email, link=url)
    if not vectorstore:
        return {"message": "No vectorstore found for the provided email and URL."}
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>----STARTED")
    
    state={
        "query": request.query,
        "context": "",
        "enhanced_query": None,
        "retrieved_docs": None,
        "recent_convo": None,
        "semantic_memory": None,
        "try_count": 0,
        "is_accurate": True,
        "evaluation": None,
        "webSearch": None,
        "vectorstore": vectorstore,
        "error": None,  
    }
    
    graph= build_langgraph()
    answer = graph.invoke(state)
    # answer =graph.invoke(state, config=RunnableConfig(recursion_limit=3))
    # if not answer:
    #     return {"message": "No answer found for the provided query."}

    return {"message": "Query answered successfully", "answer": answer["context"]}

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     print("Validation Error:", exc.json())
#     return JSONResponse(status_code=422, content={"detail": exc.errors()})

