import gradio as gr
import os
from dotenv import load_dotenv

from llama_index.llms.databricks import Databricks
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from pinecone import Pinecone
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank


# Load environment variables
load_dotenv()

# API keys setup
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Setup Databricks model for LLM
Settings.llm = Databricks(
    model="databricks-meta-llama-3-1-70b-instruct",
    api_key=DATABRICKS_TOKEN,
    api_base="https://adb-7215147325717155.15.azuredatabricks.net/serving-endpoints",
)

# Setup embedding model
Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

# Setup Pinecone vector store
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("sidindex2")
vector_index = PineconeVectorStore(pinecone_index=pinecone_index, text_key="text")

# Setup Cohere reranker
cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=3)

# Initialize the query engine with the vector store and reranker
query_engine = VectorStoreIndex.from_vector_store(vector_index).as_query_engine(
    similarity_top_k=6, node_postprocessors=[cohere_rerank]
)

new_qa_tmpl = (
    "En te basant sur le context ci-dessous relatif à la thèse du Dr Sid Ahmed Ferroukhi sur la sécurité alimentaire en Algérie.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Compte tenu des informations contextuelles et non des connaissances préalables"
    "Répond à la requête suivante de manière bien structurée et détaillée,"
    "en fournissant le maximum d'informations possibles sur le contexte ?" 
    "Veuillez également indiquer clairement la section du document ou du texte"
    "d'où provient l'extrait que vous utilisez pour votre réponse, en précisant le numéro de la section" 
    "(par exemple : section 1.1.1)"
    "Query: {query_str}\n"
    "Answer: "
)


query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": new_qa_tmpl}
)

# Modify the response generator to utilize the vector store query engine
def generate_response(message: str, history: list):
    """ 
    Use vector store query engine to search for relevant information and generate a response
    """
    response = query_engine.query(message)  
    return response.response

# Define Gradio chatbot interface
chatbot = gr.ChatInterface(
    generate_response,
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
