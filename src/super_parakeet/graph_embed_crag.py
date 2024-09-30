import cassio
import json
from typing import Dict, List, Tuple, Set, Generator

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from ragstack_knowledge_store.graph_store import CONTENT_ID
from langchain_core.graph_vectorstores.links import add_links, get_links, Link
from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
from langchain_openai.embeddings import OpenAIEmbeddings

from keyphrase_vectorizers import KeyphraseCountVectorizer


import tiktoken
from dotenv import load_dotenv
from urllib.parse import urldefrag
from tqdm import tqdm

from super_parakeet.link_extractor import LinkExtractionConverter

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

load_dotenv()

token_counter = tiktoken.encoding_for_model(EMBEDDING_MODEL)
def token_count(text: str) -> int:
    return len(token_counter.encode(text))

markdown_splitter = MarkdownTextSplitter(
    length_function = token_count,
    chunk_size = 2000,
    chunk_overlap = 0,
)

def normalize_newlines(text):
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text

def load_chunks(jsonl_file_path: str) -> Generator[List[Document], None, None]:
    html_link_extractor = LinkExtractionConverter()

    vectorizer = KeyphraseCountVectorizer(stop_words="english")
    keybert_link_extractor = KeybertLinkExtractor(
        extract_keywords_kwargs={
            "vectorizer": vectorizer,
            "use_mmr":True,
            "diversity": 0.7
        }
    )

    count = 0

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            count += 1
            if count < 3433 + 4943:
                continue
            html_link_extractor.reset()
            json_object = json.loads(line.strip())  # Parse the JSON

            url = json_object["page_url"]
            title = json_object["page_name"]
            html = json_object["page_result"]
            # snippet = json_object["page_snippet"]

            soup = BeautifulSoup(html, "html.parser")
            markdown = html_link_extractor.convert_soup(soup)

            parent_doc = Document(
                page_content=normalize_newlines(markdown),
                metadata={ CONTENT_ID: url, "title": title},
            )
            chunked_docs = markdown_splitter.split_documents([parent_doc])

            batch1: List[Document] = []
            for chunked_doc in chunked_docs:
                html_links, chunked_doc.page_content = html_link_extractor.extract_links(chunked_doc.page_content)
                html_links.add(Link.incoming(kind="hyperlink", tag=urldefrag(url).url))
                add_links(chunked_doc, html_links)
                batch1.append(chunked_doc)

            batch2: List[Document] = []
            keybert_links_batch = keybert_link_extractor.extract_many(batch1)
            for keybert_links, chunked_doc in zip(keybert_links_batch, batch1):
                pruned_links = [link for link in keybert_links if " " in link.tag]
                add_links(chunked_doc, pruned_links)
                batch2.append(chunked_doc)

            yield batch2

def load_and_insert_chunks(jsonl_file_path: str, dry_run: bool = True):
    in_links = set()
    out_links = set()
    bidir_links: Dict[str, int] = {}

    if not dry_run:
        cassio.init(auto=True)
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        graph_store = CassandraGraphVectorStore(
            embedding=embedding_model,
            keyspace="crag_graph_store"
        )

    pbar = tqdm(total=9848-3433-4943, desc="Processing", unit="html file")

    for chunk_batch in load_chunks(jsonl_file_path):
        pbar.update()
        if not dry_run:
            graph_store.add_documents(chunk_batch)

        for chunk in chunk_batch:
            links = get_links(chunk)
            for link in links:
                if link.direction == "in":
                    in_links.add(link.tag)
                elif link.direction == "out":
                    out_links.add(link.tag)
                elif link.direction == "bidir":
                    if link.tag in bidir_links:
                        bidir_links[link.tag] += 1
                    else:
                        bidir_links[link.tag] = 0
    pbar.close()

    with open("debug_links.json", "w") as f:
        json.dump(fp=f, obj={
            "in_links": list(in_links),
            "out_links": list(out_links),
            "bidir_links": bidir_links,
        })

    print(f"Links In: {len(in_links)}, Out: {len(out_links)}, BiDir: {len(bidir_links)}")

if __name__ == "__main__":
    load_and_insert_chunks("datasets/crag/task_1/html_documents.jsonl", dry_run=False)
