from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from command_gpt.utils.custom_stream import CustomStreamCallback

load_dotenv()

# Paid OpenAI API (https://platform.openai.com/account/api-keys)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

WORKSPACE_DIR = "_gpt_workspace"

default_llm_open_ai = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.2,
    max_tokens=2500,
    streaming=True,
    callbacks=[CustomStreamCallback()]  # Sets up output stream with colors
)
