from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings

def ingestion_knowledge_base(videoId):
    video_id= videoId

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
        print(transcript_list)
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print(transcript)
    except TranscriptsDisabled:
      print("No caption available for this video")
      return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def Retrieval_from_vector_store(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return docs


def augmentation_prompt(query, docs):
    prompt_template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
    {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    context = "\n".join([doc.page_content for doc in docs])
    return prompt.format(context=context, question=query)



