# CommandGPT

## Introduction
CommandGPT is an autonomous LLM system designed to carry out tasks and generate output autonomously. It's designed to be simple yet flexible, with a built in mechanism for generating robust rulesets from unstructured requests.

While the general flow is somewhat similar to [LangChain's AutoGPT](https://python.langchain.com/en/latest/use_cases/autonomous_agents/autogpt.html), the only response format enforced by this project is a command line represented at the end of responses with:
```
```cmd> { "command_name": {...}}```
```

The system tends to stick to this format well after an initial few confused responses.

## Installation
1. Set up API keys for [OpenAI](https://platform.openai.com/account/api-keys) & [Google Search](https://github.com/hwchase17/langchain/blob/master/docs/modules/agents/tools/examples/google_search.ipynb) 
2. Install the required packages using
`pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and update the placeholder values with your API keys.
4. Run the program with `python -m main` or `python main.py`

## Prompting
There are 2 main prompting mechanisms in this project. 

**To automate tasks other than research, you'll need to tweak ruleset_generator.py.**

### Ruleset Generator (ruleset_generator.py)
This file contains methods that prompt the user for input and generate a ruleset on that input through the following process (more info on this process [here](https://medium.com/@Jstnwrds55/a-prompt-template-for-generating-autogpt-input-summaries-with-chatgpt-a98388059673)):
1. Contexutalize a new LLM on how CommandGPT works through a prompt.
2. Ask the LLM for "...an input summary that will instruct CommandGPT to {ruleset_that_will}: {topic} where `ruleset_that_will` can be tweaked in the code, and `topic` is provided as user input.
3. Compose final prompt & send to LLM, which returns a detailed ruleset for use in the autonomous system.

By default, `prompt_for_research_ruleset()` is used, which prompts the LLM for "... an input summary that will instruct CommandGPT to "conduct research and generate reports on": {user_input}"

### prompt.py & prompt_generator.py
These files compose the prompt provided to the autonomous system with the ruleset, instructions, memory, etc.

`prompt_generator.py` handles composing the instructions prompt & contains static methods for building sections from other data models, for example:
- `_generate_numbered_list_section("Prime Directives", [...])` -> "`Prime Directives \n\n1...\n2...`".
- `_generate_commands_from_tools(list[BaseTool])` -> `list(str)` of formatted commands for the prompt.

`prompt.py` composes the instructions prompt into the full prompt (with context, memory,etc).

## Utils
`console_logger.py` is used for colorful logging to the console, along with coloring LLM streams

`custom_stream.py` contains a `Callbacks` class that can be passed to an LLM to automatically color the output stream when `streaming=True` 

`command_parser.py` handles parsing the command from the response, which should end with ```cmd> {....}```

`tools.py` provides an easy way to get bundled tools initialized & set up with these utils.

`evaluate.py` currently only contains a method returning stats about the current output folder
