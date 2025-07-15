from scrapy.utils.reactor import install_reactor
install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.signalmanager import dispatcher
from rag_scrap.rag_scrap.spiders.doc_scrap import RecursiveInMemorySpider

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

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

import spacy
import re
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.chrome.service import Service

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
    email: str

# def web_scraping(state:State)-> State:
#     pass

def extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc

def is_internal(url, base_netloc):
    return urlparse(url).netloc == base_netloc

# def web_scraping(state:State)-> list[dict]:
#     items = []
#     domain_doc= extract_domain(state['url'])
#     dispatcher.connect(lambda item, response, spider: items.append(item),
#                        signal=signals.item_scraped)

#     process = CrawlerProcess()
#     process.crawl(RecursiveInMemorySpider,start_url=state["url"])

#     def spider_closed(spider):
#         items.clear()
#         items.extend(spider.get_scraped_docs())

#     dispatcher.connect(spider_closed, signal=signals.spider_closed)
    
#     process.start()  
    
    
#     return items

# def scrape_recursive(start_url, max_pages=5):
#     visited = set()
#     scraped_data = []
#     queue = [start_url]
#     base_netloc = urlparse(start_url).netloc

#     while queue and len(visited) < max_pages:
#         url = queue.pop(0)
#         if url in visited:
#             continue
#         visited.add(url)

#         try:
#             response = requests.get(url, timeout=5)
#             soup = BeautifulSoup(response.text, "html.parser")

#             texts = [t.strip() for t in soup.stripped_strings]
#             full_text = "\n".join(texts)
#             if full_text:
#                 scraped_data.append({
#                     "url": url,
#                     "text": full_text,
#                 })

#             # Follow links
#             for link in soup.find_all("a", href=True):
#                 next_url = urljoin(url, link["href"])
#                 if is_internal(next_url, base_netloc) and next_url not in visited:
#                     queue.append(next_url)

#         except Exception as e:
#             print(f"[!] Error scraping {url}: {e}")
    
#     return scraped_data

def init_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)

# def scrape_recursive(start_url, max_pages=5, use_js=True, headless=True):
    visited = set()
    scraped_data = []
    queue = [start_url]
    base_netloc = urlparse(start_url).netloc

    driver = init_driver(headless) if use_js else None

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            if use_js:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                html = driver.page_source
            else:
                resp = requests.get(url, timeout=5)
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")
            full_text = "\n".join(t.strip() for t in soup.stripped_strings)
            scraped_data.append({"url": url, "text": full_text})

            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_netloc and next_url not in visited:
                    queue.append(next_url)

            time.sleep(1)  

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    if driver:
        driver.quit()

    return scraped_data

# def scrape_recursive(start_url, max_pages=40, use_js=True, headless=True):

    visited = set()
    scraped_data = []
    queue = [start_url]
    base_netloc = urlparse(start_url).netloc

    driver = init_driver(headless) if use_js else None

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            if use_js:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                html = driver.page_source
            else:
                resp = requests.get(url, timeout=5)
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup.find_all(['h1','h2','h3','h4','h5','h6']):
                tag.decompose()

            for a in soup.find_all('a', href=True):
                a.decompose()

            full_text = "\n".join(s.strip() for s in soup.stripped_strings)
            scraped_data.append({"url": url, "text": full_text})

            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_netloc and next_url not in visited:
                    queue.append(next_url)

            time.sleep(1)

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    if driver:
        driver.quit()

    return scraped_data


# def scrape_recursive(start_url, max_pages=40, use_js=True, headless=True):
    visited = set()
    scraped_data = []
    queue = [start_url]
    base_netloc = urlparse(start_url).netloc

    driver = init_driver(headless) if use_js else None

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        # print(f"Scraping: {url}")

        try:
            if use_js:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
                )
                html = driver.page_source
            else:
                resp = requests.get(url, timeout=5)
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")

            # ✅ Step 1: Extract links to queue before removing <a> tags
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_netloc and next_url not in visited:
                    queue.append(next_url)

            # ✅ Step 2: Remove <a> tags to exclude from text
            for a_tag in soup.find_all('a'):
                a_tag.decompose()

            # Optional: remove headings
            for tag in soup.find_all(['h1','h2','h3','h4','h5','h6']):
                tag.decompose()

            # ✅ Step 3: Extract cleaned text
            full_text = "\n".join(s.strip() for s in soup.stripped_strings)
            scraped_data.append({"url": url, "text": full_text})

            time.sleep(1)

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    if driver:
        driver.quit()

    return scraped_data


# def scrape_recursive(start_url, max_pages=1, use_js=True, headless=True):
    visited = set()
    scraped_data = []
    queue = [start_url]
    base_netloc = urlparse(start_url).netloc

    driver = init_driver(headless) if use_js else None

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            if use_js:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
                )
                html = driver.page_source
            else:
                resp = requests.get(url, timeout=5)
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")

            # ✅ Step 1: Extract links before removing <a> tags
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_netloc and next_url not in visited:
                    queue.append(next_url)

            # ✅ Step 2: Remove unwanted layout elements
            for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
                tag.decompose()

            # ✅ Step 3: Remove <a> and heading tags
            for tag in soup.find_all(['a']):
                tag.decompose()

            # ✅ Step 4: Focus only on <main> or <article> or fallback to <body>
            content_section = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div", class_="content") or
                soup.body
            )

            # ✅ Step 5: Extract cleaned visible text
            full_text = "\n".join(s.strip() for s in content_section.stripped_strings)
            scraped_data.append({"url": url, "text": full_text})

            time.sleep(1)

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    if driver:
        driver.quit()

    return scraped_data


# def scrape_recursive(start_url, max_pages=40, use_js=True, headless=True):
    
    visited = set()
    scraped_data = []
    queue = [start_url]
    base_netloc = urlparse(start_url).netloc

    driver = init_driver(headless) if use_js else None

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            if use_js:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                html = driver.page_source
            else:
                resp = requests.get(url, timeout=5)
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")

            # Step 1: Extract and queue valid internal links before removing <a>
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_netloc and next_url not in visited:
                    queue.append(next_url)

            # Step 2: Remove layout clutter
            for tag in soup(["header", "footer", "nav", "aside", "script", "style", "noscript"]):
                tag.decompose()

            # Step 3: Prefer main content block
            content_section = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div",class_="content") or
                soup.find("div") or
                soup.body
            )

            if not content_section:
                continue

            # Step 4: Extract cleaned visible text without stripping semantic tags
            full_text = "\n".join(s.strip() for s in content_section.stripped_strings if s.strip())

            if full_text:
                scraped_data.append({"url": url, "text": full_text})

            time.sleep(1)

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    if driver:
        driver.quit()

    return scraped_data


# def scrape_recursive(start_url, max_pages=1, use_js=True, headless=True):
#     from urllib.parse import urlparse, urljoin
#     from bs4 import BeautifulSoup, Tag
#     from selenium.webdriver.common.by import By
#     from selenium.webdriver.support.ui import WebDriverWait
#     from selenium.webdriver.support import expected_conditions as EC
#     import time, requests

#     visited = set()
#     scraped_data = []
#     queue = [start_url]
#     base_netloc = urlparse(start_url).netloc

#     driver = init_driver(headless) if use_js else None

#     while queue and len(visited) < max_pages:
#         url = queue.pop(0)
#         if url in visited:
#             continue
#         visited.add(url)
#         print(f"Scraping: {url}")

#         try:
#             # === Get HTML ===
#             if use_js:
#                 driver.get(url)
#                 WebDriverWait(driver, 10).until(
#                     EC.presence_of_element_located((By.TAG_NAME, "body"))
#                 )
#                 html = driver.page_source
#             else:
#                 resp = requests.get(url, timeout=5)
#                 html = resp.text

#             soup = BeautifulSoup(html, "html.parser")

#             # === Extract links ===
#             for link in soup.find_all("a", href=True):
#                 next_url = urljoin(url, link["href"])
#                 if urlparse(next_url).netloc == base_netloc and next_url not in visited:
#                     queue.append(next_url)

#             # === Remove layout junk ===
#             for tag in soup(["header", "footer", "nav", "aside", "script", "style", "noscript"]):
#                 tag.decompose()

#             # === Focus on main content area ===
#             content_section = (
#                 soup.find("main") or
#                 soup.find("article") or
#                 soup.find("div", class_="blog-post") or
#                 soup.find("div", class_="content") or
#                 soup.body
#             )
#             if not content_section:
#                 continue

#             # === Group content under headings ===
#             current_heading = "Untitled Section"
#             current_chunk = []
#             page_chunks = []

#             for elem in content_section.descendants:
#                 if isinstance(elem, Tag):
#                     if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
#                         # Save previous section
#                         if current_chunk:
#                             chunk_text = f"{current_heading.strip()}\n{'-'*50}\n" + "\n".join(current_chunk).strip()
#                             page_chunks.append(chunk_text)
#                             current_chunk = []
#                         current_heading = elem.get_text(strip=True)

#                     elif elem.name in ["p", "li", "blockquote"]:
#                         text = elem.get_text(strip=True)
#                         if text:
#                             current_chunk.append(text)

#             # Add final chunk
#             if current_chunk:
#                 chunk_text = f"{current_heading.strip()}\n{'-'*50}\n" + "\n".join(current_chunk).strip()
#                 page_chunks.append(chunk_text)

#             # Merge chunks into one page entry
#             full_text = "\n\n---\n\n".join(page_chunks)

#             if full_text.strip():
#                 scraped_data.append({"url": url, "text": full_text})

#             time.sleep(1)

#         except Exception as e:
#             print(f"[!] Error scraping {url}: {e}")

#     if driver:
#         driver.quit()

#     return scraped_data



# def scrape_recursive(start_url, max_pages=30, use_js=True, headless=True):
    from urllib.parse import urlparse, urljoin
    from bs4 import BeautifulSoup, Tag
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time, requests

    visited = set()
    scraped_data = []
    queue = [start_url]
    base_netloc = urlparse(start_url).netloc

    driver = init_driver(headless) if use_js else None

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            # === Fetch HTML ===
            if use_js:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                html = driver.page_source
            else:
                resp = requests.get(url, timeout=5)
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")

            # === Extract internal links ===
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_netloc and next_url not in visited:
                    queue.append(next_url)

            # === Remove unwanted tags ===
            for tag in soup(["header", "footer", "nav", "aside", "script", "style", "noscript"]):
                tag.decompose()

            # === Focus on main content area ===
            content_section = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div", class_="blog-post") or
                soup.find("div", class_="content") or
                soup.body
            )
            if not content_section:
                continue

            # === Chunk by heading ===
            current_heading = "Untitled Section"
            current_chunk = []
            page_chunks = []

            for elem in content_section.descendants:
                if isinstance(elem, Tag):
                    tag_name = elem.name.lower()

                    # Headings define new chunks
                    if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        if current_chunk:
                            chunk_text = f"{current_heading.strip()}\n{'-'*50}\n" + "\n".join(current_chunk).strip()
                            page_chunks.append(chunk_text)
                            current_chunk = []
                        current_heading = elem.get_text(strip=True)

                    # Text blocks we want to include
                    elif tag_name in ["p", "li", "blockquote"]:
                        text = elem.get_text(strip=True)
                        if text:
                            current_chunk.append(text)

                    # Code blocks (either inside <pre> or standalone <code>)
                    elif tag_name in ["pre", "code"]:
                        code_text = elem.get_text()
                        if code_text:
                            current_chunk.append("```" + code_text.strip() + "```")

            # Add final chunk
            if current_chunk:
                chunk_text = f"{current_heading.strip()}\n{'-'*50}\n" + "\n".join(current_chunk).strip()
                page_chunks.append(chunk_text)

            # === Combine with separators ===
            full_text = "\n\n---\n\n".join(page_chunks)

            if full_text.strip():
                scraped_data.append({"url": url, "text": full_text})

            time.sleep(1)

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    if driver:
        driver.quit()

    return scraped_data

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



# # def scrape_recursive(start_url, max_pages=30, headless=True):  
#     visited = set()
#     queue = [start_url]
#     scraped_data = []
#     base = urlparse(start_url).netloc

#     while queue and len(visited) < max_pages:
#         url = queue.pop(0)
#         if url in visited:
#             continue

#         doc = fetch_clean_page(url, headless=headless)
#         if not doc:
#             continue

#         text = doc.page_content
#         visited.add(url)
#         scraped_data.append(doc)

#         driver = init_driver(headless)
#         driver.get(url)
#         WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
#         for a in driver.find_elements(By.TAG_NAME, "a"):
#             href = a.get_attribute("href")
#             if href and urlparse(href).netloc == base and href not in visited:
#                 queue.append(href)
#         driver.quit()

#         time.sleep(1)

#     return scraped_data


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
    transcript=YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
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

# def doc_ingestion(items:list[dict],email:str,url:str)->None:


#     if not items:
#         return

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = []

#     for item in items:
#         # doc = Document(page_content=item['text'], metadata={"url": item['url']})
#         documents.extend(text_splitter.split_documents([item]))

#     collection_name = safe_collection_name(email,url)
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
#     vector_store = get_vectorstore(email=email, link=url,force_recreate=True)
    
#     BATCH_SIZE = 10
#     for i in range(0, len(documents), BATCH_SIZE):
#         batch = documents[i:i + BATCH_SIZE]
#         try:
#             vector_store.add_documents(batch)
#         except Exception as e:
#             print(f"Failed to add batch {i // BATCH_SIZE + 1}: {e}")
    
#     return


# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings



# def doc_ingestion(items, email: str, url: str) -> None:
#     if not items:
#         return

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = []

#     # for item in items:
#     #     text = item.get("page_content") or item.get("text")  # support both keys
#     #     if not text or not text.strip():
#     #         continue

#     #     metadata = item.get("metadata", {})
#     #     # Fallback to top-level keys if metadata not nested
#     #     metadata = {
#     #         "url": metadata.get("source") or item.get("url"),
#     #         "title": metadata.get("title") or item.get("title"),
#     #         "description": metadata.get("description") or item.get("description"),
#     #     }

#     #     doc = Document(page_content=text.strip(), metadata=metadata)
#     #     documents.extend(splitter.split_documents([doc]))

#     # if not documents:
#     #     return
#     for item in items:
#         data=splitter.split_text(item)
#         documents.extend(data)
    
#     if not documents:
#         return
#     vector_store = get_vectorstore(email=email, link=url, force_recreate=True)
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#     BATCH_SIZE = 10
#     for i in range(0, len(documents), BATCH_SIZE):
#         batch = documents[i : i + BATCH_SIZE]
#         try:
#             vector_store.add_documents(batch)
#         except Exception as e:
#             print(f"Failed to add batch {i // BATCH_SIZE + 1}: {e}")

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
