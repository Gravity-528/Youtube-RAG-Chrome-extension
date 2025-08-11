import time
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser,Document
from langchain.llms import OpenAI
from typing import TypedDict
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import LLMChain 
from collections import defaultdict, Counter
from langchain.vectorstores.qdrant import Qdrant
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from memory.main import base_config
import copy
from mem0 import Memory
from langchain_openai import ChatOpenAI
from typing import TypedDict, Union, List, Any,Literal
import os
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langfuse import observe
from qdrant_client import models
# from langchain.output_parsers import StrOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from openai import OpenAI
from langchain.load import dumps,loads
import json
from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    search_depth="basic",
    time_range="day"
)

client= OpenAI()

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9c926bfc-d2f9-4af7-8cf9-63e000e62143" 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-cb7fc817-99a2-4857-a2ea-280070bfdf7e" 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,             
)

langfuse_handler = CallbackHandler()
def get_config_for_user(email: str):
    collection_name = email.replace("@", "_at_").replace(".", "_dot_")

    config = copy.deepcopy(base_config)

    config["vector_store"]["config"]["collection_name"] = collection_name

    return config

# web_search_tool = DuckDuckGoSearchRun()
class QueryState(TypedDict):
    query: str
    context: str
    enhanced_query: Union[List[str], None]
    retrieved_docs: Union[List[Document], None]
    recent_convo: Union[List[dict], None]
    semantic_memory: Union[List[Document], None]
    try_count: int
    is_accurate: bool
    evaluation: int
    webSearch: Union[str, None]
    vectorstore: Qdrant
    error: Union[str, None]  


def query_enhancer(state: QueryState) -> QueryState:
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>----ENHANCER")
    prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    **Provide these alternative questions in a Json array**. Original query: {query}
    Output:
    """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(query=state["query"])
    state["enhanced_query"] = output
    return state


from collections import defaultdict
from langchain.schema import Document

from collections import defaultdict
import hashlib

def reciprocal_rank_fusion(results: list[list], k=60):
    scores = defaultdict(float)
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "").strip()
            title = doc.metadata.get("title", "").strip()
            desc = doc.metadata.get("description", "").strip()
            combined = f"{source} | {title} | {desc}".strip(" | ")
            key = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            scores[key] += 1 / (rank + k)
            doc_map[key] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in fused]

@observe(name="multiquery_search_and_reranking")
def multiquery_search_and_reranking(state: QueryState, llm=None) -> QueryState:
    embeddings = OpenAIEmbeddings()
    vs = state["vectorstore"]
    all_queries = state["enhanced_query"]
    
    # print("started embeddings process------------------------------------------------------------------------------>")
    resp = embeddings.client.create(
      model="text-embedding-ada-002",
      input=all_queries
    )
    # print(resp)
    # vectors = [item["embedding"] for item in resp["data"]]
    vectors = [item.embedding for item in resp.data]
    # vectors = [embeddings.embed_query(q) for q in all_queries]

    requests = [
        models.SearchRequest(vector=vec, limit=5, with_payload=True)
        for vec in vectors
    ]
    print("started searching process------------------------------------------------------------------------------>")
    responses:list[list] = vs.client.search_batch(
        collection_name=vs.collection_name,
        requests=requests,
    )

    retrieved_lists = [
        [
            Document(
                page_content=pt.payload.get("page_content", ""),
                metadata={**pt.payload, "id": pt.id}
            )
            for pt in resp
        ]
        for resp in responses
    ]

    main_query_docs = vs.similarity_search(state["query"], k=5)
    retrieved_lists.append(main_query_docs)
    # print("finished searching process------------------------------------------------------------------------------>")
    if not any(retrieved_lists):
        state["error"] = "No documents retrieved."
        return state

    # print("Retrieved Lists:-------------->", retrieved_lists)
    fused = reciprocal_rank_fusion(retrieved_lists, k=60)
    top_docs = [doc for doc, _ in fused[:5]]
    state["retrieved_docs"] = top_docs
    return state

@observe(name="get_answer")
def get_answer(state: QueryState, llm: Any = None) -> QueryState:
    # print(">>> ANSWER GENERATION")
    if state["try_count"] > 4:
        # print(">>> Aborting jnjchbjhbrc.")
        return state
    
    state["try_count"] = state.get("try_count", 0) + 1
    if not state.get("retrieved_docs"):
        state["context"] = "No relevant documents found."
        return state

    query = state["query"]
    
    output_parts = []
    for entry in state["retrieved_docs"]:
        if not isinstance(entry, list):
            continue
    
        page_content = title = desc = ""
    
        i = 0
        while i < len(entry):
            field_pair = entry[i]
            if isinstance(field_pair, list) and len(field_pair) == 2:
                field_name, value = field_pair
                if field_name == "page_content":
                    page_content = value or ""
                elif field_name == "title":
                    title = value or ""
                elif field_name == "description":
                    desc = value or ""
            i += 2 
    
        parts = []
        if title.strip():
            parts.append(f"Title: {title.strip()}")
        if desc.strip():
            parts.append(f"Description: {desc.strip()}")
        if page_content.strip():
            parts.append(page_content.strip())
    
        if parts:
            output_parts.append("\n".join(parts))
    
    # retrieved_docs = "\n\n---\n\n".join(output_parts)
    retrieved_docs=state["retrieved_docs"]

    # print("Retrieved Docs:----------------------------------------------------------------------------------------->", retrieved_docs)


    recent_convo= state.get("recent_convo", [])
    # recent_convo = "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in state.get("recent_convo", []))
    # recent_convo= "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in state.get("recent_convo", [])) if state.get("recent_convo") else ""
    # semantic_memory =
    # semantic_memory = "\n\n".join(doc.page_content for doc in state.get("semantic_memory", []))
    semantic_memory=""
    webSearch = state.get("webSearch", "")

    # Combine system instructions and context into one message
    system_prompt = f"""You are an expert assistant tasked with answering the user's query using the provided context. 
    Use the **Retrieved Documents** as the primary source of truth use description and page content present in metadata. If available and relevant, incorporate information from the **Recent Conversation** (last 5 turns between the user and assistant) and **Semantic Memory** (long-term context and prior interactions).
    
    Guidelines:
    - go through all the document take refernece from source,title,description and page content of the document.If any of the retrieved documents provide relevant information, generate a clear, accurate, and concise answer based strictly on their content. Answer in simple and easy to understand language keeping in mind you are explaining to a learner who doesn't know about the answer but also stick to the context given. Also return the most important url of all the documents present in the context in new paragraph with reference:<Url>.
    - If all the documents are not relevant or do not address the query, respond with: **"I don't know"**.
    - If helpful, use the recent conversation and semantic memory to add clarity or fill in missing context, but do not hallucinate.
    - First, assess the relevance of the documents. If they are relevant, summarize the key points related to the query and then answer. If not, say: **"Not relevant"**.
    - If **webSearch** is not empty take reference from it
    ---
    
    **Query:**  
    {query}
    
    **Retrieved Documents:**  
    {retrieved_docs}
    
    **Recent Conversation:**  
    {recent_convo}
    
    **Semantic Memory:**  
    {semantic_memory}
    
    **Web Search Results:**  
    {webSearch}
    
    Your response:
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    # state["context"] = response["choices"][0]["message"]["content"].strip()
    state["context"] = response.choices[0].message.content.strip()
    return state

        
@observe(name="evaluate_answer")
def evaluate_answer(state: QueryState) -> QueryState:
    if not state.get("context"):
        state["context"] = "No answer generated."
        raise ValueError("No context available for evaluation.")

    prompt=PromptTemplate(
        input_variables=["query","response"],
        template="""
        You are an impartial evaluator and be neutral and no sugarcoating.You are Given with query and a response. Your job is to rate how relevant and helpful a given response is in addressing the given query. 

        Instructions:
        1. Carefully compare the response to the given original query.
        2. Consider factual alignment, topical relevance, and how well the response satisfies the query intent.
        3. Then, assign a score from 1 to 10 based on the rubric below.

        ### Scoring Rubric:
        Score 1-3: Poor — The response is mostly irrelevant, off-topic, or incorrect.
        Score 4-6: Fair — Somewhat relevant, partially addresses the query, but lacks clarity or depth.
        Score 7-8: Good — Mostly relevant and correct, but could be more complete or focused.
        Score 9-10: Excellent — Fully relevant, accurate, and addresses the query thoroughly.
        
       **Only return a Json Object with field score which will be between 1-10 based on scoring rubric**

        ### Query:
        {query}
        
        ### Response:
        {response}
        
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(query=state["query"], response=state["context"])
    try:
        output_json = json.loads(output)
        state["evaluation"] = output_json["score"]
    except Exception as e:
        print(f"Error parsing output: {output}. Exception: {e}")
        state["evaluation"] = 0
    return state

@observe(name="webSearch")
def query_upgradation_andSearch(state: QueryState, llm=None) -> QueryState:
    if not state.get("evaluation") or not isinstance(state["evaluation"], int):
        # print("type--------------------------------------------------------------->",type(state["evaluation"]))
        state["query"] = "No evaluation available."
        return state
    time.sleep(3)
    # search=web_search_tool.run(state['query'])
    try:
       search = search_tool.invoke({"query": state['query']})
       state["webSearch"] = search
    # print("Web Search Results:----------------------------------------------------------------------------------------->", search)
    except Exception as e:
        print(f"Error during web search: {e}")
        search = "No results found or an error occurred during the web search."
    
    time.sleep(1)
    return state

from langgraph.graph.state import Command
from typing import Literal

# @observe(name="is_accuracy_good")
def is_accuracy_good(state: QueryState) -> Literal["query_upgradation_andSearch", "other_part"]:
    score = state.get("evaluation", 0)
    tc = state.get("try_count", 0)

    # print("try count---------------------------------------------------------------------------------------------------------------->", tc)

    if score > 5 or tc >= 3:
        return "other_part"

    return "query_upgradation_andSearch"

@observe(name="other_part")
def other_part(state: QueryState) -> QueryState:
    return state
   
def build_langgraph() -> StateGraph:
    graph_builder = StateGraph(QueryState)
    
    graph_builder.add_node("query_enhancer", query_enhancer)
    graph_builder.add_node("multiquery_search_and_reranking", multiquery_search_and_reranking)
    graph_builder.add_node("get_answer", get_answer)
    graph_builder.add_node("evaluate_answer", evaluate_answer)
    graph_builder.add_node("query_upgradation_andSearch", query_upgradation_andSearch)
    graph_builder.add_node("other_part", other_part)

    graph_builder.add_edge(START, "query_enhancer")
    graph_builder.add_edge("query_enhancer", "multiquery_search_and_reranking")
    graph_builder.add_edge("multiquery_search_and_reranking", "get_answer")
    graph_builder.add_edge("get_answer", "evaluate_answer")
    graph_builder.add_conditional_edges(
    "evaluate_answer",                      
    is_accuracy_good,
   )
    graph_builder.add_edge("query_upgradation_andSearch", "get_answer")
    graph_builder.add_edge("other_part", END)
    
    return graph_builder.compile().with_config({"callbacks": [langfuse_handler]})