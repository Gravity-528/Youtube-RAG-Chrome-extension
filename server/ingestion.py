import os
from scrapy.utils.reactor import install_reactor
install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# from scrapy import signals
# from scrapy.crawler import CrawlerProcess
# from scrapy.signalmanager import dispatcher
# from rag_scrap.rag_scrap.spiders.doc_scrap import RecursiveInMemorySpider

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, RequestBlocked

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from vectorstore import get_vectorstore
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

# import spacy
import re
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.chrome.service import Service
from youtube_transcript_api.proxies import WebshareProxyConfig

# nlp = spacy.load("xx_sent_ud_sm")

def clean_text(text):
    return text.replace("\n", " ").replace("[Music]", "").strip()

def fetch_transcript(video_id, preferred_languages=["hi", "en"]):
    ytt_api = YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username=os.getenv("PROXY_USERNAME"),
            proxy_password=os.getenv("PROXY_PASSWORD"),
        )
    )
    try:
        fetched = ytt_api.fetch(video_id, languages=preferred_languages)
        return fetched.to_raw_data()
    except TranscriptsDisabled:
        print("No captions available for this video")
        return []
    except RequestBlocked:
        print("Request blocked, possibly due to rate limiting or IP issues")
        return []


from langchain.schema import Document

# def chunk_by_multilingual_sentences_with_timestamps(transcript_list, chunk_size=1000, chunk_overlap=200):
#     full_text = ""
#     time_map = []
#     for entry in transcript_list:
#         clean = clean_text(entry["text"])
#         if not clean:
#             continue
#         offset = len(full_text)
#         full_text += clean + " "
#         time_map.append((offset, entry["start"]))

#     chunks = []
#     i = 0
#     text_len = len(full_text)

#     while i < text_len:
#         end = min(text_len, i + chunk_size)
#         chunk_text = full_text[i:end].strip()

#         ts = 0.0
#         for offset, start in time_map:
#             if offset <= i:
#                 ts = start
#             else:
#                 break

#         chunks.append(Document(page_content=chunk_text, metadata={"start_time": ts}))

#         i = end - chunk_overlap

#     return chunks

def chunk_by_multilingual_sentences_with_timestamps(transcript_list, chunk_size=1000, chunk_overlap=200):
    chunks = []
    current_chunk = ""
    current_start = None
    idx = 0

    try:
        while idx < len(transcript_list):
            current_chunk = ""
            current_start = transcript_list[idx]["start"]
            i = idx
    
            # Build chunk up to 1000 characters
            while i < len(transcript_list) and len(current_chunk) + len(transcript_list[i]["text"]) + 1 <= 1000:
                current_chunk += transcript_list[i]["text"].strip() + " "
                i += 1
    
            # Save the chunk
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={"start": current_start}
            ))
    
            # Move index forward for next chunk with 200 character overlap
            # We re-calculate how many characters to backtrack
            if i == len(transcript_list):
                break  # no more to process
    
            # Start at the first subtitle entry that gives an overlap of ~200 chars
            overlap_chars = 0
            idx = i - 1
            while idx > 0 and overlap_chars < 200:
                overlap_chars += len(transcript_list[idx]["text"]) + 1
                idx -= 1
            idx += 1  # 

    except Exception as e:
        print(f"Error while chunking transcript: {e}")

    return chunks

class ScrapIt(TypedDict):
    url:str
    text:str

class State(TypedDict):
    video_id: str
    url:str
    list_to_scrap: list[ScrapIt]
    type: str 
    email: str


def extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc

def is_internal(url, base_netloc):
    return urlparse(url).netloc == base_netloc

def init_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)


def fetch_clean_page(url, headless=True):
    loader = SeleniumURLLoader(
        urls=[url],
        headless=headless,
        browser="chrome",
        arguments=["--headless=new"],
        continue_on_failure=True
    )
    docs = loader.load()  
    return docs[0] if docs else None


def scrape_recursive(start_url, max_pages=30, headless=True):
    visited = set()
    queue = [start_url]
    scraped_data = []
    base = urlparse(start_url).netloc

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue

        doc = fetch_clean_page(url, headless=headless)
        if not doc:
            continue

        text = doc.page_content
        visited.add(url)
        scraped_data.append(doc)

        driver = init_driver(headless=headless)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        for a in driver.find_elements(By.TAG_NAME, "a"):
            href = a.get_attribute("href")
            if href and urlparse(href).netloc == base and href not in visited:
                queue.append(href)
        driver.quit()

        time.sleep(1)

    return scraped_data


def youtube_ingestion(video_id: str, email: str)->None:
    # transcript=YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
    transcript=fetch_transcript(video_id,languages=['en', 'hi'])
    if not transcript:
        raise TranscriptsDisabled("No captions available for this video")
    pass
    
    chunks = chunk_by_multilingual_sentences_with_timestamps(transcript)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = get_vectorstore(email=email, link=video_id, force_recreate=True)

    # vector_store.from_documents(
    #     documents=chunks,
    #     collection_name=f'state["video_id"] for transcript',
    #     embeddings=embeddings
    # )
    vector_store.add_documents(chunks)
    return

def safe_collection_name(email: str, link: str) -> str:
    email_part = email.replace('@', '_').replace('.', '_')
    
    link_part = re.sub(r'https?://', '', link)  
    link_part = re.sub(r'[^a-zA-Z0-9_]', '_', link_part)  

    return f"user_{email_part}_{link_part}"


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

def doc_ingestion(items: list, email: str, url: str) -> None:
    if not items:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []

    # items are already Documents, so split them properly
    for doc in items:
        try:
            split_docs = splitter.split_documents([doc])
            chunks.extend(split_docs)
        except Exception as e:
            print(f"Splitter error for doc {doc.metadata.get('source', doc.metadata.get('url', ''))}: {e}")

    if not chunks:
        return

    vector_store = get_vectorstore(email=email, link=url, force_recreate=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # LangChain handles embeddings internally in add_documents

    BATCH_SIZE = 10
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"Failed to add batch {i // BATCH_SIZE + 1}: {e}")
