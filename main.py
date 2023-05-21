# This file initializes and starts a CommandGPT instance with a custom stream callback.

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import faiss

from config import default_llm_open_ai
from command_gpt.utils.tools import get_base_tools
from command_gpt.utils.custom_stream import CustomStreamCallback
from command_gpt.prompting.ruleset_generator import prompt_for_research_ruleset
from command_gpt.command_gpt import CommandGPT

# See config.py for API key setup and default LLMs
llm_open_ai = default_llm_open_ai

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Prepare LLM with streaming for live output
llm = ChatOpenAI(
    temperature=0.8, 
    streaming=True, 
    verbose=True,
    callbacks=[CustomStreamCallback()]
)

ruleset = prompt_for_research_ruleset()

# Initialize CommandGPT with tools, LLM, and memory
command_gpt = CommandGPT.from_ruleset_and_tools(
    ruleset,
    tools=get_base_tools(),
    llm=llm,
    memory=vectorstore.as_retriever()
)

# Run CommandGPT
command_gpt.run()
