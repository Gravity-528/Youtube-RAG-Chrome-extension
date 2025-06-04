from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

def ingestion_knowledge_base(video_id):
    transcript_list = fetch_transcript(video_id)
    if not transcript_list:
        return None

    chunks = chunk_by_multilingual_sentences_with_timestamps(transcript_list)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def format_docs(retrieved_docs):
    result = ""
    for doc in retrieved_docs:
        ts = doc.metadata.get("start_time", 0.0)
        mins = int(ts // 60)
        secs = int(ts % 60)
        timestamp = f"[{mins:02d}:{secs:02d}]"
        result += f"{timestamp} {doc.page_content}\n\n"
    return result.strip()

def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """You are a helpful assistant. Use the following context from a YouTube video transcript to answer the user's question. Mention relevant timestamps if helpful.
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
        return "No transcript available for this video."

    chain = build_qa_chain(vector_store)
    answer = chain.invoke(query)
    return answer
