# CommandGPT

## Introduction
CommandGPT is an autonomous LLM system designed to carry out tasks and generate output autonomously. It's designed to be simple yet flexible, with a core loop that operates off an initial ruleset combined with a built-in mechanism for generating robust rulesets from user input.

This project uses [LangChain](https://github.com/hwchase17/langchain) to drive LLM interactions. To get the primary Agent to use LangChain [tools](https://python.langchain.com/en/latest/modules/agents/tools/getting_started.html) consistently, it is prompted to provide commands using openining/closing tags and a command line format:
```
<cmd> command_name --arg1 value1 --arg2 value2 </cmd>
```

Because the system is designed using LangChain's [Tool](https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html) class for building the commands list, Tools can be added and removed painlessly, and the commands list will be handled gracefully in the prompt generator. Commands are mapped from Tools to the following format:
```
search: Gather search results from query (they will automatically be written to a file called 'results_{query}'), arguments: --tool_input <string> ()
write_file: Write file to disk, arguments: --file_path <string> (name of file), --text <string> (text to write to file), --append <boolean> (Whether to append to an existing file.)
...
```
which the AI is able to understand & use consistently.

## Installation
1. Set up API keys for [OpenAI](https://platform.openai.com/account/api-keys) & [Google Search](https://github.com/hwchase17/langchain/blob/master/docs/modules/agents/tools/examples/google_search.ipynb) 
2. Install the required packages using
`pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and update the placeholder values with your API keys.
4. Run the program with `python -m main` or `python main.py`

## Prompting
**There are 2 main prompting mechanisms in this project.** 
### **Ruleset Generator (ruleset_generator.py)**
This file contains an Agent Class with various constructors for starting the ruleset generation chat. This process is similar to using AIPRM to generate input summaries (which is briefly covered [here](https://medium.com/@Jstnwrds55/a-prompt-template-for-generating-autogpt-input-summaries-with-chatgpt-a98388059673)):

Out of the box this is set up to prompt an LLM to "...generate a ruleset that instructs CommandGPT to {search the web, read research papers, and project future trends on}: {user_input}" with an interactive chat for refining the ruleset.

You'll need to tinker with `main.py` to use this beyond the boilerplate, but there is ample in-code documentation and it's pretty simple once the {request}: {topic} convention makes sense.

**Once you have a ruleset you like, you can paste it into `custom_ruleset` in `main.py`**

Generally, Ruleset Generation works like this
1. Contexutalize a new LLM on how CommandGPT works through a prompt.
2. Ask the LLM for "...an input summary that will instruct CommandGPT to {ruleset_that_will}: {topic}, which allows for flexible meta-prompting.
3. Compose final prompt & send to LLM, which returns a detailed ruleset for use in the autonomous system.
4. Interactive chat process allows feedback for improving prompt (such as removing search, focusing on certain aspects, epmhasizing file writing, etc.)

**Experimental: You can provide feedback to refine your ruleset, but it can be finnicky so be concise and specific for best results.**

### **CommandGPT Prompt (prompt.py & prompt_generator.py**)
These files compose the prompt provided to the autonomous system with the ruleset, instructions, memory, etc.

`prompt_generator.py` handles composing the instructions prompt & contains static methods for building sections from other data models, for example:
- `_generate_numbered_list_section("Prime Directives", [...])` -> "`Prime Directives \n\n1...\n2...`".
- `_generate_commands_from_tools(List[BaseTool])` -> `List[str]` of formatted commands for the prompt.

`prompt.py` composes the instructions prompt into the full prompt (with context, memory,etc).


## Tooling
`tools.py` defines some custom tools for specific use cases such as writing search results and manually handling new line characters

`toolkits.py` defines a `BaseToolkit` class that provides all tools in a way that's easy to subclass (for defining toolkits with certain tools removed)

## Utils
`console_logger.py` is used for colorful logging to the console, along with coloring LLM streams

`custom_stream.py` contains a `Callbacks` class that can be passed to an LLM to automatically color the output stream when `streaming=True` 

`command_parser.py` handles parsing the command from the response, which should include `<cmd> command_name --arg1 value1 --arg2 value2 </cmd>`

`evaluate.py` currently only contains a method returning stats about the current output folder
