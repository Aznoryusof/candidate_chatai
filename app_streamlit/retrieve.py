import os
import sys

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI

load_dotenv()

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)

DB_PATH = os.environ.get('DB_PATH')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
OPEN_AI_BASE=os.environ.get('OPEN_AI_BASE')
API_KEY=os.environ.get('API_KEY')
SEED=int(os.environ.get('SEED'))
TEMPERATURE=float(os.environ.get('TEMPERATURE'))
TOP_K=int(os.environ.get('TOP_K'))


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


def build_chain():
    # Define the embedding function used for documents
    embedding_function = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,

    )

    # Instantiate the document vector DB from the path
    db = Chroma(
        persist_directory=os.path.join(MAIN_DIR, DB_PATH), 
        embedding_function=embedding_function
    )
    
    # Instantiate the llm chat client
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=OPEN_AI_BASE,
        temperature=TEMPERATURE
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


if __name__ == "__main__":
    #prompt = "Hi, can you tell me more about Aznor and how he can contribute as an AI Platform Engineer?"
    prompt = "Hi, give me a summary of Aznor's work experience in the past three years in bullet points. It is now 2023."
    #prompt = "Hi, can you tell me more about Elon Musk?"
    #prompt = "Hi, how many world cups have singapore won in football?"

    chain = build_chain()
    output = run_chain(chain, prompt, [{"role": "user", "content": "Hi, my name is Vincent."}, {"role": "assistant", "content": "Hi Vincent. Pleasure to meet you."}])
    print(output["answer"], end="\n\n")
    print(output["source_documents"])