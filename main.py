# This file initializes and starts a CommandGPT instance with a custom stream callback.
# A custom ruleset can be provided, or one can be generated using the RulesetGeneratorAgent.

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import faiss
from command_gpt.prompting.ruleset_generator import RulesetGeneratorAgent

from config import default_llm_open_ai
from command_gpt.tooling.toolkits import BaseToolkit, MemoryOnlyToolkit
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

placeholder_ruleset = """
You are ECO-gpt, an AI designed to search the web, read research papers, and project future trends on ecosystem deterioration, climate change, and the future of humanity. 

To achieve this, you will first search the web for relevant articles and research papers on ecosystem deterioration, climate change, and the future of humanity. You will then read and analyze these sources to identify key trends, patterns, and insights related to the topic. 

Using this information, you will project future trends and potential outcomes related to ecosystem deterioration, climate change, and the future of humanity. You will write detailed markdown (.md) files summarizing your findings, including any relevant statistics, graphs, or visualizations. 

Throughout this process, you will ensure that your content is accurate, well-researched, and up-to-date. You will also strive to provide a comprehensive overview of the topic, covering both the current state of affairs and potential future developments.
"""

# Add your custom ruleset here
custom_ruleset = """
...paste ruleset here...
"""

# endregion
# region Ruleset Generation
# - Generate a ruleset for CommandGPT (e.g. "CommandGPT is...Generate a ruleset for CommandGPT that will {request}: {topic}")
# USAGE - This can be used with varying levels of user input:
# - RulesetGeneratorAgent.from_request_and_topic() generates a ruleset from a hardcoded request and topic
# - RulesetGeneratorAgent.from_request_prompt_for_topic() prompts user for topic (e.g. "Generate a ruleset for CommandGPT that will {request provided in constructor}: {topic}")
# - RulesetGeneratorAgent.from_empty_prompt_for_request_and_topic() prompts user for request and topic (e.g. "Generate a ruleset for CommandGPT that will {request}: {topic}")

# To initialize CommandGPT with a custom ruleset, paste it above, NOT HERE
if custom_ruleset != """
...paste ruleset here...
""":
    current_ruleset = custom_ruleset
else:  # If no custom ruleset is specified above
    # If true, prompts user for topic, otherwise uses hardcoded topic.
    should_prompt_user = True

    if should_prompt_user:  # Prompt user for topic for ruleset generation
        ruleset_generator = RulesetGeneratorAgent.from_request_prompt_for_topic(
            request="search the web, read research papers, and project future trends on",
            # Prompt displayed to user for input
            prompt_for_topic="predict future trends on",
            llm=llm
        )
        current_ruleset = ruleset_generator.run()  # Interactive, accepts user feedback
    else:
        # Generate ruleset from hard coded values
        ruleset_generator_no_input = RulesetGeneratorAgent.from_request_and_topic(
            request="search the web, read research papers, and project future trends on",
            topic="ecosystem deterioration, climate change, and the future of humanity",
            llm=llm
        )
        # Interactive, accepts user feedback
        current_ruleset = ruleset_generator_no_input.run()

# endregion
# region CommandGPT Initialization

# Prepare toolkits
base_toolkit = BaseToolkit()  # Contains all tools
no_web_toolkit = MemoryOnlyToolkit()  # Contains all tools except search/web

# Initialize CommandGPT with tools, LLM, and memory
command_gpt = CommandGPT.from_ruleset_and_tools(
    current_ruleset,
    tools=base_toolkit.get_toolkit(),
    llm=llm,
    memory=vectorstore.as_retriever()
)

# Run CommandGPT
command_gpt.run()

# endregion
