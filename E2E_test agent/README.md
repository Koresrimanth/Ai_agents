
# E2E Testing Agent using GenAI + Playwright

This project uses a GenAI agent to convert natural language instructions into automated Playwright-based end-to-end (E2E) tests.

## Features
- LangChain + LangGraph agent workflow
- Playwright automation of browsers
- Wikipedia and Weather tools supported
- Custom instruction parsing with assertion handling

## Setup Instructions
1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```

2. Run the agent:
```bash
python agent/main.py
```

## Folder Structure
- `agent/main.py` - Main agent logic and workflow
- `agent/utils.py` - Utility functions for tool and prompt integration
- `tests/sample_app.py` - Sample Flask web app for testing
- `README.md` - Overview and setup instructions
