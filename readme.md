# CommandGPT

## Introduction
CommandGPT is an autonomous LLM system designed to carry out tasks and generate output autonomously. It's designed to be simple yet flexible, with a built in mechanism for generating robust rulesets from unstructured requests.

This project uses [LangChain](https://github.com/hwchase17/langchain) to drive LLM interactions. The LLM is instructed to provide a command line at the end of it's response formatted as:
```
<cmd> command_name --arg1 value1 --arg2 value2 </cmd>
```

This system is designed to use LangChain's [Tool](https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html) class for building the commands list. Tools can be added and removed painlessly, and the commands list will be handled gracefully in the prompt generator.

## Installation
1. Set up API keys for [OpenAI](https://platform.openai.com/account/api-keys) & [Google Search](https://github.com/hwchase17/langchain/blob/master/docs/modules/agents/tools/examples/google_search.ipynb) 
2. Install the required packages using
`pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and update the placeholder values with your API keys.
4. Run the program with `python -m main` or `python main.py`

## Prompting
There are 2 main prompting mechanisms in this project. 

### Ruleset Generator (ruleset_generator.py)
This file contains an Agent Class that has various constructors that assist in starting the ruleset generation chat. This process is similar to using AIPRM to generate input summaries (which is briefly covered [here](https://medium.com/@Jstnwrds55/a-prompt-template-for-generating-autogpt-input-summaries-with-chatgpt-a98388059673)):

Currently, you'll need to tinker with `main.py` to use this beyond the boilerplate. There is ample in-code documentation.

Generally, the ruleset generator works like this
1. Contexutalize a new LLM on how CommandGPT works through a prompt.
2. Through various constructors, allow user input for the {ruleset_that_will}, {topic}, or both
3. Ask the LLM for "...an input summary that will instruct CommandGPT to {ruleset_that_will}: {topic}.
4. Compose final prompt & send to LLM, which returns a detailed ruleset for use in the autonomous system.
5. Interactive chat process allows feedback for improving prompt (such as removing search, focusing on certain aspects, epmhasizing file writing, etc.)

By default, the request is for blog posts and the topic is user provided, so again, tinker with `main.py` to tweak this.

### prompt.py & prompt_generator.py
These files compose the prompt provided to the autonomous system with the ruleset, instructions, memory, etc.

`prompt_generator.py` handles composing the instructions prompt & contains static methods for building sections from other data models, for example:
- `_generate_numbered_list_section("Prime Directives", [...])` -> "`Prime Directives \n\n1...\n2...`".
- `_generate_commands_from_tools(List[BaseTool])` -> `List[str]` of formatted commands for the prompt.

`prompt.py` composes the instructions prompt into the full prompt (with context, memory,etc).

## Utils
`console_logger.py` is used for colorful logging to the console, along with coloring LLM streams

`custom_stream.py` contains a `Callbacks` class that can be passed to an LLM to automatically color the output stream when `streaming=True` 

`command_parser.py` handles parsing the command from the response, which should end with ```cmd> {....}```

`tools.py` has some custom tooling and provides an easy way to get bundled tools initialized & set up with these utils.

`evaluate.py` currently only contains a method returning stats about the current output folder
