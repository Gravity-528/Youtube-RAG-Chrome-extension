from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser,Document
from langchain.llms import OpenAI
from typing import TypedDict
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import LLMChain 
from collections import defaultdict, Counter
from langchain.vectorstores.qdrant import Qdrant

from typing import TypedDict, Union

class QueryState(TypedDict):
    query: str
    context: str
    enhanced_query: Union[str, None]
    retrieved_docs: Union[list[Document], None]
    recent_convo: Union[list[dict], None]
    semantic_memory: Union[list[Document], None]
    try_count: int


def query_enhancer(state: QueryState, llm) -> QueryState:
    prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are an intelligent and neutral query enhancer. 
    Given a user query, generate 3-5 semantically and conceptually related queries that stay on topic and either rephrase, extend, or add useful angles. 
    These queries will be used to search a large knowledge base. Be concise and focused. Avoid repeating the same sentence structure.
    Respond in valid JSON list format like: ["...", "...", "..."].
    
    Example:
    query: Why is my JWT token expiring too early?
    Output:
    [
        "What controls the expiration time of a JWT token?",
        "Can clock drift cause early JWT expiration?",
        "How do I set the correct exp field in JWT tokens?",
        "Why does my JWT token become invalid before expiry?",
        "How to implement long-lived sessions using refresh tokens?"
    ]
    
    Example:
    query: Attention mechanism in transformers
    Output:
    [
        "What is the role of attention in transformer models?",
        "How does scaled dot-product attention work?",
        "What are key, query, and value vectors in transformers?",
        "How is multi-head attention different from single-head?",
        "How does attention enable context handling in NLP tasks?"
    ]
    
    Example:
    query: How to handle failed payments in Stripe?
    Output:
    [
        "How do I detect a failed payment in Stripe Checkout?",
        "What webhook events indicate a failed Stripe transaction?",
        "How to notify users of failed payments in Stripe billing?",
        "What are common reasons for payment failure in Stripe?",
        "How to retry failed payments automatically using Stripe?"
    ]
    
    Example:
    query: Add Google login to Next.js
    Output:
    [
        "How to implement Google OAuth2 login in a Next.js app?",
        "How to use NextAuth.js for Google authentication in Next.js?",
        "What are the environment variables needed for Google login in Next.js?",
        "How to handle callbacks and redirect URIs in Google OAuth with Next.js?",
        "How to secure API routes in Next.js after Google login?"
    ]
    
    Example:
    query: "Docker is an open-source containerization platform by which you can pack your application and all its dependencies into a standardized unit called a container." Explain its meaning
    Output:
    [
        "What does it mean that Docker is an open-source platform?",
        "What is containerization in the context of software development?",
        "What does Docker 'pack' into a container — what exactly gets bundled?",
        "What is a Docker container, and how is it a standardized unit?",
        "Why is using containers like Docker beneficial for running applications?"
    ]
    
    Query: {query}
    Output:
    """
   )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(query=state["query"])
    state["enhanced_query"] = output
    return state

def multiquery_search_and_reranking(state: QueryState, llm=None) -> QueryState:
    embeddings = OpenAIEmbeddings()
    vectorstore: Qdrant = state["vectorstore"] 
    main_query = state["query"]
    enhanced_queries = state["enhanced_query"]

    all_queries = [main_query] + enhanced_queries
    query_embeddings = embeddings.embed_documents(all_queries)

    retrieved_all = []

    
    for query_emb in query_embeddings:
        docs = vectorstore.similarity_search_by_vector(query_emb, k=5)
        retrieved_all.extend(docs)

    
    content_counter = defaultdict(list)  
    for doc in retrieved_all:
        content_counter[doc.page_content].append(doc)

    ranked = sorted(content_counter.items(), key=lambda x: len(x[1]), reverse=True)

    top_docs = [group[1][0] for group in ranked[:6]] 

    state["retrieved_docs"] = top_docs
    return state

def get_answer(state: QueryState, llm=None) -> QueryState:
    if not state.get("retrieved_docs"):
        state["context"] = "No relevant documents found."
        return state

    retrieved_texts = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])

    recent_convo = state.get("recent_convo", "")
    semantic_memory = state.get("semantic_memory", "")

    prompt = PromptTemplate(
        input_variables=["query", "retrieved_docs", "recent_convo", "semantic_memory"],
        template="""You are an expert assistant tasked with answering the user's query using the provided context. 
        Use the **Retrieved Documents** as the primary source of truth. If available and relevant, incorporate information from the **Recent Conversation** (last 5 turns between the user and assistant) and **Semantic Memory** (long-term context and prior interactions).
        
        Guidelines:
        - If the retrieved documents provide relevant information, generate a clear, accurate, and concise answer based strictly on their content.
        - If the documents are not relevant or do not address the query, respond with: **"I don't know"**.
        - If helpful, use the recent conversation and semantic memory to add clarity or fill in missing context, but do not hallucinate.
        - First, assess the relevance of the documents. If they are relevant, summarize the key points related to the query and then answer. If not, say: **"Not relevant"**.
        
        ---
        
        **Query:**  
        {query}
        
        **Retrieved Documents:**  
        {retrieved_docs}
        
        **Recent Conversation:**  
        {recent_convo}
        
        **Semantic Memory:**  
        {semantic_memory}
        
        ---
        
        Your response:
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(
        query=state["query"],
        retrieved_docs=retrieved_texts,
        recent_convo=recent_convo,
        semantic_memory=semantic_memory
    )

    state["context"] = output
    return state

def evaluate_answer(state: QueryState, llm=None) -> QueryState:
    if not state.get("context"):
        state["context"] = "No answer generated."
        return state

    prompt=PromptTemplate(
        input_variables=["query","response"],
        template="""
        You are an impartial evaluator. Your job is to rate how relevant and helpful a given response is in addressing a user query. 
        
        Instructions:
        1. Carefully compare the response to the original query.
        2. Consider factual alignment, topical relevance, and how well the response satisfies the query intent.
        3. Provide a short, specific explanation of your evaluation.
        4. Then, assign a score from 1 to 10 based on the rubric below.
        
        ### Query:
        {query}
        
        ### Response:
        {response}
        
        ### Scoring Rubric:
        Score 1-3: Poor — The response is mostly irrelevant, off-topic, or incorrect.
        Score 4-6: Fair — Somewhat relevant, partially addresses the query, but lacks clarity or depth.
        Score 7-8: Good — Mostly relevant and correct, but could be more complete or focused.
        Score 9-10: Excellent — Fully relevant, accurate, and addresses the query thoroughly.
        
        Format:
        Feedback: <your brief analysis>  
        [RESULT] <integer from 1 to 10>
        """
        )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(query=state["query"], response=state["context"])

    state["evaluation"] = output
    return state

# def query_upgradation(state:QueryState) -> QueryState:
    