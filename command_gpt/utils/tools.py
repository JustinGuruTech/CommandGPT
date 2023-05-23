# Defines custom LangChain tools & Bundles tools set up with custom stream callback for printing to console

from pathlib import Path
import re
from typing import Optional

from langchain import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.tools.human.tool import HumanInputRun
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)

from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, WORKSPACE_DIR
from command_gpt.utils.custom_stream import CustomStreamCallback

WORKSPACE_PATH = Path(WORKSPACE_DIR)


class WriteFileToolNewlines(WriteFileTool):
    def _run(
        self,
        file_path: str,
        text: str,
        append: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Replace any occurrences of (\\x2+)n with \n
        text = re.sub(r'\\+n', '\n', text)
        return super()._run(file_path, text, append, run_manager)


class SearchAndWriteTool:
    """
    Runs a search query, writes the results to a file, and returns a result message.
    """

    def __init__(self, search: GoogleSearchAPIWrapper):
        self.search = search

    def run(self, query: str) -> str:
        """Run a search query, write the results to a file, and return a result message."""
        # Run the search query
        results = self.search.run(query)
        # Sanitize the query to use it as a filename
        safe_query = self.sanitize_filename(query)
        # Prepare the filename
        file_name = f"search_results/results_{safe_query}.txt"
        file_path = WORKSPACE_PATH / file_name
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the results to the file
        with open(file_path, "w") as file:
            file.write(results)
        # Return a result message
        return f"Search results:\n\n {results}\n\n written to file named: {file_name}"

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Remove any character from name that is not alphanumeric, space, underscore or hyphen.
        """
        return re.sub(r'[^a-zA-Z0-9 _-]', '', name)


# region Bundled Tools

def get_base_tools():
    """
    Gets a list of tools set up with custom stream callback
    """

    # Set up custom google search with automatic writing of results
    search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
    )
    search_and_write = SearchAndWriteTool(search)
    search_tool = Tool(
        name="search",
        func=search_and_write.run,
        description="Gather search results from query (they will automatically be written to a file called 'results_\{query\}).",
        callbacks=[CustomStreamCallback()]
    )
    # Regular LangChain tools with custom stream callback
    write_file_tool = WriteFileToolNewlines(
        root_dir=WORKSPACE_DIR,
        callbacks=[CustomStreamCallback()]
    )
    read_file_tool = ReadFileTool(
        root_dir=WORKSPACE_DIR,
        callbacks=[CustomStreamCallback()]
    )
    list_directory_tool = ListDirectoryTool(
        root_dir=WORKSPACE_DIR,
        description="List files to read from or append to.",
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

# endregion
