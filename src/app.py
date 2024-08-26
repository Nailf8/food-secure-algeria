import gradio as gr
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pinecone import Pinecone
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.llms.databricks import Databricks
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.memory import ChatMemoryBuffer
from scripts.pipeline import ChatPipeline, ResponseWithChatHistory

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

from scripts.router import RouterOutputParser


# ----------------------------------------------- Env Setup ----------------------------------------------- #

# Load environment variables
load_dotenv()

# API keys setup
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# ---------------------------------------------- Index Setup ---------------------------------------------- #

# Setup embedding model
Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Setup Pinecone vector store
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("sidindex")
index = PineconeVectorStore(pinecone_index=pinecone_index, text_key="text")


# ------------------------------------------ Chat Pipeline Setup ------------------------------------------ #

# First, we create an input component to capture the user query
input_component = InputComponent()

# Next, we use the LLM to rewrite a user query
rewrite = (
    "Please write a query to a semantic search engine using the current conversation.\n"
    "\n"
    "\n"
    "{chat_history_str}"
    "\n"
    "\n"
    "Latest message: {query_str}\n"
    'Query:"""\n'
)
rewrite_template = PromptTemplate(rewrite)

llama3 = Databricks(
    model="databricks-meta-llama-3-1-70b-instruct",
    api_key=DATABRICKS_TOKEN,
    api_base="https://adb-7215147325717155.15.azuredatabricks.net/serving-endpoints",
)

# using that, we will retrieve...
retriever = VectorStoreIndex.from_vector_store(index).as_retriever(similarity_top_k=15)

# then postprocess/rerank with Cohere reranker
reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=10)

# and finally generates the response using this component
response_component = ResponseWithChatHistory(
    llm=llama3,
    system_prompt=(
        "You are a Q&A system. You will be provided with the previous chat history, "
        "as well as possibly relevant context, to assist in answering a user message."
    ),
)

# pipeline builder
pipeline = ChatPipeline()
pipeline.add_components(input_component, rewrite_template, llama3, retriever, reranker, response_component)

# buffer to handle memory
pipeline_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

# ------------------------------------------- SQL Agent Setup ------------------------------------------- #

# Get SQLite database from URI
db = SQLDatabase.from_uri("sqlite:///demo.db")

# Here we use GPT 3.5 
gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Agent creation
sql_agent = create_sql_agent(
    gpt35, db = db, agent_type = "openai-tools", verbose = True
)


# ---------------------------------------------- Router Setup ------------------------------------------- #

choices = [
    "Utile pour répondre aux questions sur les politiques de soutien, l'innovation et les scénarios prospectifs.",
    "Utile pour répondre aux questions sur des données statistiques precises."
]

choices_str = "\n\n".join([f"{idx+1}. {choice}" for idx, choice in enumerate(choices)])

router_prompt_template = PromptTemplate(
    "Some choices are given below. It is provided in a numbered list (1 to"
    " {num_choices}), where each item in the list corresponds to a summary.\n"
    "---------------------\n{context_list}\n---------------------\nUsing only the choices"
    " above and not prior knowledge, return the top choice that is most relevant"
    " to the question: '{query_str}'\n"
)

output_parser = RouterOutputParser()


def router_query(message: str, pipeline, agent):
    fmt_prompt = router_prompt_template.format(
        num_choices=len(choices),
        context_list=choices_str,
        query_str=message,
        max_outputs=1
    )
    fmt_json_prompt = output_parser.format(fmt_prompt)
    raw_output = llama3.complete(fmt_json_prompt)
    parsed = output_parser.parse(str(raw_output))
    
    if parsed.choice == 1:
        return str(pipeline._run(message, pipeline_memory)).replace("assistant:", "").strip()
    elif parsed.choice == 2:
        return str(agent.invoke(message)["output"])


# ------------------------------------------ Gradio Integration ------------------------------------------ #

# Create a wrapper function that Gradio will use
def generate_response_wrapper(message: str, history : list):
    return router_query(message, pipeline=pipeline, agent=sql_agent)

# Define Gradio chatbot interface
chatbot = gr.ChatInterface(
    generate_response_wrapper,
    chatbot=gr.Chatbot(
        avatar_images=["images/user.jpg", "images/chatbot.png"],
        height="64vh"
    ),
    title="Food Secure Algeria 2050",
    description="Feel free to ask any question.",
    theme="soft",
    submit_btn="⬅ Send",
    retry_btn="🔄 Regenerate Response",
    undo_btn="↩ Delete Previous",
    clear_btn="🗑️ Clear Chat"
)

# Launch the Gradio chatbot
chatbot.launch()