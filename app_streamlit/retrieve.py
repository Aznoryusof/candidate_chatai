import os
import sys

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI

load_dotenv()

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)

DB_PATH = os.environ.get('DB_PATH')
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL')
EMBEDDINGS_PATH = os.environ.get('EMBEDDINGS_PATH')
OPEN_AI_BASE=os.environ.get('OPEN_AI_BASE')
API_KEY=os.environ.get('API_KEY')
SEED=int(os.environ.get('SEED'))
TEMPERATURE=float(os.environ.get('TEMPERATURE'))
TOP_K=int(os.environ.get('TOP_K'))
ES_URL = os.environ.get('ES_URL')


def _get_chat_history(inputs):
    # Processes the chat history by combining the responses from 
    # user and the assistant and returning as a formatted string 
    res = []
    for _i in inputs:
        if _i.get("role") == "user":
            user_content = _i.get("content")
        if _i.get("role") == "assistant":
            assistant_content = _i.get("content")
            res.append(f"user:{user_content}\nassistant:{assistant_content}")
    return "\n".join(res)


def build_chain(embedding_function=None, callbacks=None):
    # Define the embedding function used for documents
    if embedding_function is None:
        embedding_function = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            cache_folder=EMBEDDINGS_PATH
        )

    # Instantiate the document vector DB from the path
    db = ElasticsearchStore(
        es_url=ES_URL,
        index_name="candidate_index",
        embedding=embedding_function,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(
            # hybrid=True, <- This is a paid feature of ES 
        )
    )
    
    # Instantiate the llm chat client
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=OPEN_AI_BASE,
        temperature=TEMPERATURE,
        streaming=True,
        callbacks=callbacks
    )

    # Specify the variables and template to 
    condensed_template = """
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    condense_question_prompt = PromptTemplate.from_template(
        condensed_template
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
        condense_question_prompt=condense_question_prompt,
        return_source_documents=True,
        get_chat_history=_get_chat_history
    )

    return qa


def run_chain(chain, prompt, history=[]):
    return chain({"question": prompt, "chat_history": history})
