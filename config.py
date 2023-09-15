# This file sets up the API keys and default language models for the examples.
# It includes both OpenAI and HuggingFace Language Models.

from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI

from CommandGPT.command_gpt.utils.custom_stream import CustomStreamCallback

load_dotenv()

# Paid OpenAI API (https://platform.openai.com/account/api-keys)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Free HuggingFace API (https://huggingface.co/settings/tokens)
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

WORKSPACE_DIR = "_gpt_workspace"

# region Instantiate Language Models
# - Different models can be used for different results/use cases
# - Temperature - 0-1: "randomness/diversity" of output (higher = more random)

# Paid OpenAI model (https://openai.com/blog/openai-api)
default_llm_open_ai = ChatOpenAI(
    # model_name="gpt-4",
    temperature=0.3,
    # max_tokens=2500,
    streaming=True,
    callbacks=[CustomStreamCallback()]  # Sets up output stream with colors
)

# endregion
