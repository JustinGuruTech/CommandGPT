# Bundles tools set up with custom stream callback

from langchain import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.tools.human.tool import HumanInputRun

from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, WORKSPACE_DIR
from command_gpt.utils.custom_stream import CustomStreamCallback

def get_base_tools():
    """
    Gets a list of tools set up with custom stream callback
    """
    search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
    )

    search_tool = Tool(
        name="search",
        func=search.run,
        description="Search the web with SerpApi",
        callbacks=[CustomStreamCallback()]
    )
    write_file_tool = WriteFileTool(
        root_dir=WORKSPACE_DIR, 
        callbacks=[CustomStreamCallback()]
    )
    read_file_tool = ReadFileTool(
        root_dir=WORKSPACE_DIR,
        callbacks=[CustomStreamCallback()]
    )
    list_directory_tool = ListDirectoryTool(
        root_dir=WORKSPACE_DIR,
        callbacks=[CustomStreamCallback()]
    )

    finish_tool = Tool(
        name="finish",
        func=lambda: None,
        description="End the program.",
        callbacks=[CustomStreamCallback()]
    )

    human_input_tool = Tool(
        name="human_input",
        func=HumanInputRun,
        description="Get input from a human if you find yourself overly confused.",
        callbacks=[CustomStreamCallback()]
    )

    base_tools = [
        search_tool,
        write_file_tool,
        read_file_tool,
        list_directory_tool,
        human_input_tool,
        finish_tool,
    ]

    return base_tools

