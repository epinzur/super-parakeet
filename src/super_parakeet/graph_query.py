import cassio
from dotenv import load_dotenv
from langchain_community.graph_vectorstores.cassandra import CassandraGraphVectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
KEYSPACE_NAME = "hotpot_graph_store"

load_dotenv()


def get_llm(chat_model_name: str) -> BaseChatModel:
    return ChatOpenAI(model=chat_model_name)


def get_graph_store(keyspace_name: str) -> CassandraGraphVectorStore:
    cassio.init(auto=True)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return CassandraGraphVectorStore(embedding=embedding_model, keyspace=keyspace_name)


def query_pipeline(depth: int, search_type: str, **kwargs):
    llm = get_llm(chat_model_name=CHAT_MODEL)

    graph_store = get_graph_store(keyspace_name=KEYSPACE_NAME)

    retriever = graph_store.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "depth": depth,
        },
    )

    prompt_template = """
    Retrieved Information:
    {retrieved_docs}

    User Query:
    {query}

    Response Instruction:
    Please generate a response that uses the retrieved information to directly and clearly answer the user's query. Ensure that the response is relevant, accurate, and well-organized.
    """  # noqa: E501

    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        {
            "retrieved_docs": retriever,
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
