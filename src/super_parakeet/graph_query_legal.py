import cassio
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.graph_vectorstores.cassandra import CassandraGraphVectorStore
from langchain_core.graph_vectorstores import GraphVectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from typing import List, Tuple

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
KEYSPACE_NAME = "legal_graph_store"

load_dotenv()

class GraphReRankRetriever(GraphVectorStoreRetriever):
    cross_encoder = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=None)
    re_rank:bool = False

    def __init__(self, *args, re_rank: str = "true", **kwargs):
        super().__init__(*args, **kwargs)
        self.re_rank = re_rank.lower() == "true"


    def rerank_with_cross_encoder(self, query: str, documents: List[Document], k: int = 5) -> List[Document]:
        if not self.re_rank:
            return documents

        # Re-rank documents using the cross-encoder
        scored_documents = [
            (doc, self.cross_encoder(f"{query} [SEP] {doc.page_content}")[0][0]['score']) for doc in documents
        ]

        # Sort documents by score in descending order and return top 5
        ranked_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_documents[:k]]  # Return the top 5 documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        result = super()._get_relevant_documents(query, run_manager=run_manager)
        return self.rerank_with_cross_encoder(query=query, documents=result)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Call the original async method
        result = await super()._aget_relevant_documents(query, run_manager=run_manager)
        return self.rerank_with_cross_encoder(query=query, documents=result)



def get_llm(chat_model_name: str) -> BaseChatModel:
    return ChatOpenAI(model=chat_model_name, temperature=0.0)


def get_graph_store(keyspace_name: str) -> CassandraGraphVectorStore:
    cassio.init(auto=True)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return CassandraGraphVectorStore(embedding=embedding_model, keyspace=keyspace_name)


def query_pipeline(depth: int, search_type: str, re_rank: str = "true", **kwargs):
    llm = get_llm(chat_model_name=CHAT_MODEL)
    graph_store = get_graph_store(keyspace_name=KEYSPACE_NAME)

    retriever = GraphReRankRetriever(
        re_rank=re_rank,
        vectorstore=graph_store,
        search_type=search_type,
        search_kwargs={
            "k": 5,
            "fetch_k": 20,  # Fetch 20 docs, but we'll return top 5 after re-ranking
            "depth": depth,
        },
    )

    # # Prepare the retriever
    # retriever = graph_store.as_retriever(
    #     search_type=search_type,
    #     search_kwargs={
    #         "k": 5,
    #         "fetch_k": 20,  # Fetch 20 docs, but we'll return top 5 after re-ranking
    #         "depth": depth,
    #     },
    # )


    # Define the prompt template
    prompt_template = """
    Retrieved Information:
    {retrieved_docs}

    User Query:
    {query}

    Response Instruction:
    Please generate a response without using markdown that uses the retrieved information to directly and clearly answer the user's query. Ensure that the response is relevant, accurate, and well-organized.
    """  # noqa: E501

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Return the pipeline with retriever and re-ranking step
    return (
        {
            "retrieved_docs": retriever,
            "query": RunnablePassthrough(),
        }
        | prompt                        # Generate prompt with re-ranked docs
        | llm                           # Pass through the LLM
        | StrOutputParser()              # Final output parser
    )
