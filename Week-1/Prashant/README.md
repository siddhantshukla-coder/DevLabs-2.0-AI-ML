# Steam Game Discovery Agent

A CLI-based AI assistant powered locally by ollama. This agent searches the storefront, fetches information about specific games, and helps recommend games.

## Tech Stack

* **Language:** Python 3.10+
* **LLM:** Local [Ollama](https://ollama.com/)
* **Framework:** [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/)

## Setup & Installation

**1. Install Ollama and Pull the model**

```bash
ollama pull llama3.1
```

**2. Clone the repository**

```bash
git clone https://github.com/Prashant-SG14/steam-game-discovery-agent.git
cd steam-discovery-agent
```

**3. Create a venv**

```bash
python -m venv .venv
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

Start the agent by running the main script:
```bash
python agent.py
```
Type exit or quit to close the application.
