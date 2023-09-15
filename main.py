# This file initializes and starts a CommandGPT instance with a custom stream callback.

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

import faiss

from config import default_llm_open_ai
from command_gpt.tooling.toolkits import BaseToolkit, MemoryOnlyToolkit
from command_gpt.command_gpt import CommandGPT


# Define embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

# Prepare LLM with streaming for live output
llm = default_llm_open_ai

# Add your custom ruleset here
ruleset = """
Good afternoon! Could you please test out your tooling? First, using the human_input tool which prompts the user for a topic, then use test_custom_tool to provide your response. Then, finish the program. Ensure you only use one command at a time. Thank you!
"""

# Prepare toolkits
base_toolkit = BaseToolkit()  # Contains all tools
no_web_toolkit = MemoryOnlyToolkit()  # Contains all tools except search/web

# Initialize CommandGPT with tools, LLM, and memory
command_gpt = CommandGPT.from_ruleset_and_tools(
    ruleset,
    tools=base_toolkit.get_toolkit(),
    llm=llm,
    memory=vectorstore.as_retriever()
)

# Run CommandGPT
command_gpt.run()
