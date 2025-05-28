from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def fetch_transcript(video_id, preferred_languages=["hi", "en"]):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
        return transcript
    except TranscriptsDisabled:
        print("No caption available for this video")
        return []

def ingestion_knowledge_base(video_id):
    transcript_list = fetch_transcript(video_id)
    if not transcript_list:
        return None

    transcript_text = " ".join(chunk["text"] for chunk in transcript_list)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """You are a helpful assistant. Use the following context to answer the question.
{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

def ask_question(video_id, query):
    vector_store = ingestion_knowledge_base(video_id)
    if not vector_store:
        return "No vector store available."

    chain = build_qa_chain(vector_store)
    answer = chain.invoke(query)
    return answer
