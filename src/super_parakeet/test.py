import cassio
import json
from typing import Dict, List, Tuple, Set, Generator

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from ragstack_knowledge_store.graph_store import CONTENT_ID
from langchain_community.graph_vectorstores.cassandra import CassandraGraphVectorStore
from langchain_core.graph_vectorstores.links import add_links, get_links, Link
from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
from langchain_openai.embeddings import OpenAIEmbeddings

from keyphrase_vectorizers import KeyphraseCountVectorizer
from transformers import PreTrainedTokenizerFast


import tiktoken
from dotenv import load_dotenv
from urllib.parse import urldefrag
from tqdm import tqdm



vectorizer = KeyphraseCountVectorizer(stop_words="english")
keybert_link_extractor = KeybertLinkExtractor(
    extract_keywords_kwargs={
        "keyphrase_ngram_range":(1, 3),
        "use_mmr":True,
        "diversity": 0.7
    }
)

links = keybert_link_extractor.extract_one("What is the situation in Palestine?")

for link in links:
    print(link.tag)
