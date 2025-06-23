from scrapy.utils.reactor import install_reactor
install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.signalmanager import dispatcher
from scr.scr.spiders.example import RecursiveInMemorySpider

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore

import spacy

nlp = spacy.load("xx_sent_ud_sm")

def clean_text(text):
    return text.replace("\n", " ").replace("[Music]", "").strip()

def fetch_transcript(video_id, preferred_languages=["hi", "en"]):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video")
        return []

def chunk_by_multilingual_sentences_with_timestamps(transcript_list, chunk_size=1000, chunk_overlap=200):
    text_with_offsets = []
    full_text = ""
    current_offset = 0

    for entry in transcript_list:
        clean = clean_text(entry["text"])
        if not clean:
            continue
        full_text += clean + " "
        text_with_offsets.append((entry["start"], current_offset, len(clean)))
        current_offset += len(clean) + 1 

    doc = nlp(full_text.strip())
    sentence_spans = list(doc.sents)

    def find_start_time(char_index):
        for start_time, offset, length in text_with_offsets:
            if offset <= char_index < offset + length:
                return start_time
        return 0.0

    chunks = []
    buffer = ""
    buffer_len = 0
    start_time = None
    i = 0

    while i < len(sentence_spans):
        sent = sentence_spans[i].text.strip()
        char_index = sentence_spans[i].start_char
        sent_time = find_start_time(char_index)

        if start_time is None:
            start_time = sent_time

        if buffer_len + len(sent) <= chunk_size:
            buffer += " " + sent
            buffer_len += len(sent)
            i += 1
        else:
            chunks.append(Document(
                page_content=buffer.strip(),
                metadata={"start_time": start_time}
            ))
            overlap_start = max(0, i - 3)
            buffer = " ".join(sentence_spans[j].text.strip() for j in range(overlap_start, i))
            buffer_len = len(buffer)
            start_time = find_start_time(sentence_spans[overlap_start].start_char)

    if buffer.strip():
        chunks.append(Document(
            page_content=buffer.strip(),
            metadata={"start_time": start_time}
        ))

    return chunks

class ScrapIt(TypedDict):
    url:str
    text:str

class State(TypedDict):
    video_id: str
    url:str
    list_to_scrap: list[ScrapIt]
    type: str 

# def web_scraping(state:State)-> State:
#     pass

def extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc

def web_scraping(state:State)-> State:
    items = []
    domain_doc= extract_domain(state['url'])
    dispatcher.connect(lambda item, response, spider: items.append(item),
                       signal=signals.item_scraped)

    process = CrawlerProcess()
    process.crawl(RecursiveInMemorySpider,start_url=state["url"],domain=domain_doc)

    def spider_closed(spider):
        items.clear()
        items.extend(spider.get_scraped_docs())

    dispatcher.connect(spider_closed, signal=signals.spider_closed)
    
    process.start()  
    
    state['list_to_scrap'] = items
    return state

def youtube_ingestion(state:State)->State:
    transcript=YouTubeTranscriptApi.get_transcript(state['video_id'], languages=['en', 'hi'])
    if not transcript:
        raise TranscriptsDisabled("No captions available for this video")
    pass
    
    chunks = chunk_by_multilingual_sentences_with_timestamps(transcript)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store= QdrantVectorStore.from_documents(
        documents=chunks,
        collection_name=f'state["video_id"] for transcript',
        embeddings=embeddings
    )
    return state

def doc_ingestion(state:State)->State:
    if not state['list_to_scrap']:
        return state
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    
    for item in state['list_to_scrap']:
        doc = Document(page_content=item['text'], metadata={"url": item['url']})
        documents.extend(text_splitter.split_documents([doc]))
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        collection_name=f'state["url"] for web scraping',
        embeddings=embeddings
    )
    
    return state



graph_builder = StateGraph[State]()
graph_builder.add_node(START, "Start")
graph_builder.add_node(web_scraping, "Web Scraping")
graph_builder.add_node(youtube_ingestion, "YouTube Ingestion")
graph_builder.add_node(doc_ingestion, "Document Ingestion")
graph_builder.add_edge(START, web_scraping)
graph_builder.add_edge(START, youtube_ingestion)
graph_builder.add_edge(web_scraping, doc_ingestion)
graph_builder.add_edge(youtube_ingestion, END)
graph_builder.add_edge(doc_ingestion, END)
