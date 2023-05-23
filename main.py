# This file initializes and starts a CommandGPT instance with a custom stream callback.

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import faiss
from command_gpt.prompting.ruleset_generator import RulesetGeneratorAgent

from config import default_llm_open_ai
from command_gpt.utils.tools import get_base_tools
from command_gpt.utils.custom_stream import CustomStreamCallback
from command_gpt.command_gpt import CommandGPT

# region Setup

# See config.py for API key setup and default LLMs
llm_open_ai = default_llm_open_ai

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

# Prepare LLM with streaming for live output
llm = ChatOpenAI(
    temperature=0.2,
    streaming=True,
    verbose=True,
    callbacks=[CustomStreamCallback()]
)

# Add your custom ruleset here
custom_ruleset = """
...paste ruleset here...
"""

# endregion
# region Ruleset Generation
# - Generate a ruleset for CommandGPT from various levels of user input
# DEFAULT:
# - RulesetGeneratorAgent.from_request_and_topic() generates a ruleset from a hardcoded request and topic
#
# ADDITIONAL OPTIONS:
# - RulesetGeneratorAgent.from_empty_prompt_for_request_and_topic() prompts user for request and topic (e.g. "Generate a ruleset for CommandGPT that will {request}: {topic}")
# - RulesetGeneratorAgent.from_request_prompt_for_topic() prompts user for topic (e.g. "Generate a ruleset for CommandGPT that will {request provided in constructor}: {topic}")

# To initialize CommandGPT with a custom ruleset, paste it above
if custom_ruleset != """
...paste ruleset here...
""":
    current_ruleset = custom_ruleset
# otherwise, add your request & topic here to generate a ruleset
else:
    # Initialize agent for generating CommandGPT ruleset from request & topic
    # NOTE: use from_request_and_topic with the arg 'topic' instead to hardcode topic
    ruleset_generator = RulesetGeneratorAgent.from_request_prompt_for_topic(
        request="research and most importantly generate blog posts on",
        prompt_for_topic="enter a topic to generate blog posts on",
        llm=llm
    )
    # Generate ruleset (this is interactive and accepts feedback)
    current_ruleset = ruleset_generator.run()

# endregion
# region CommandGPT Initialization

# Initialize CommandGPT with tools, LLM, and memory
command_gpt = CommandGPT.from_ruleset_and_tools(
    current_ruleset,
    tools=get_base_tools(),
    llm=llm,
    memory=vectorstore.as_retriever()
)

# Run CommandGPT
command_gpt.run()

# endregion
