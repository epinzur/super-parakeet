import cassio
import json
import time
from typing import Dict, List, Generator

from langchain_core.documents import Document
from ragstack_knowledge_store.graph_store import CONTENT_ID
from langchain_core.graph_vectorstores.links import add_links, get_links
from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
from langchain_community.graph_vectorstores.cassandra import CassandraGraphVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings

from keyphrase_vectorizers import KeyphraseCountVectorizer


from dotenv import load_dotenv
from tqdm import tqdm

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
BATCH_SIZE = 250
KEYSPACE = "hotpot_graph_store"

load_dotenv()

keybert_link_extractor = KeybertLinkExtractor(
    extract_keywords_kwargs={
        "vectorizer": KeyphraseCountVectorizer(stop_words="english"),
        "use_mmr":True,
        "diversity": 0.7
    }
)

def count_lines_in_file(file_path: str) -> int:
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)

def build_document_batch(text_batch: List[str], title_batch:List[str]):
    doc_batch: List[Document] = []
    keybert_links_batch = keybert_link_extractor.extract_many(text_batch)
    for keybert_links, text, title in zip(keybert_links_batch, text_batch, title_batch):
        # drop links with one word
        pruned_links = [link for link in keybert_links if " " in link.tag]
        doc = Document(
            page_content=text,
            metadata={CONTENT_ID: title}
        )

        add_links(doc, pruned_links)
        doc_batch.append(doc)
    return doc_batch


def load_facts(jsonl_file_path: str) -> Generator[List[Document], None, None]:
    total_lines = count_lines_in_file(file_path=jsonl_file_path)

    with open(jsonl_file_path, 'r') as file:
        with tqdm(total=total_lines, desc="Reading lines") as pbar:
            title_batch: List[str] = []
            text_batch: List[str] = []

            line = file.readline()
            while line:
                pbar.update()
                json_object = json.loads(line.strip())  # Parse the JSON

                try:
                    title = json_object["fact"].encode('utf-8').decode('unicode_escape')
                except Exception:
                    title = json_object["fact"]

                try:
                    text = json_object["text"].encode('utf-8').decode('unicode_escape')
                except Exception:
                    text = json_object["text"]

                title_batch.append(title)
                text_batch.append(text)

                if len(text_batch) == BATCH_SIZE:
                    yield build_document_batch(text_batch=text_batch, title_batch=title_batch)
                    title_batch: List[str] = []
                    text_batch: List[str] = []

                # Read the next line
                line = file.readline()

            yield build_document_batch(text_batch=text_batch, title_batch=title_batch)

def init_graph_store() -> CassandraGraphVectorStore:
    cassio.init(auto=True)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return CassandraGraphVectorStore(
        embedding=embedding_model,
        keyspace=KEYSPACE,
    )



def load_and_insert_chunks(jsonl_file_path: str, dry_run: bool = True):
    in_links = set()
    out_links = set()
    bidir_links: Dict[str, int] = {}

    if not dry_run:
        graph_store = init_graph_store()

    for chunk_batch in load_facts(jsonl_file_path):
        if not dry_run:
            while True:
                try:
                    graph_store.add_documents(chunk_batch)
                    break
                except Exception as e:
                    print(f"Encountered issue trying to store document batch: {e}")
                    time.sleep(2)
                    graph_store = init_graph_store()

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

    with open("debug_links_hotpot.json", "w") as f:
        json.dump(fp=f, obj={
            "in_links": list(in_links),
            "out_links": list(out_links),
            "bidir_links": bidir_links,
        })

    print(f"Links In: {len(in_links)}, Out: {len(out_links)}, BiDir: {len(bidir_links)}")

if __name__ == "__main__":
    load_and_insert_chunks("datasets/hotpotqa/facts.jsonl", dry_run=False)
