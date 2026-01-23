# 42subject.ai
This project is a specialized tool designed for students and community members of 42 Network to interact with school subjects using Artificial Intelligence. It provides a more intuitive way to understand project requirements, explore edge cases, and clarify complex subject PDF instructions.

## Features
- AI-Powered Subject Analysis: Upload or select a 42 subject to get instant explanations of complex concepts.

- RAG-based Question Answering: Uses Retrieval-Augmented Generation to ensure answers are strictly grounded in the official subject documentation.

- Project Guidance: Get help with understanding the "Common Instructions," "Mandatory Part," and "Bonus Part" of 42 projects.

- Code Logic Support: Assistance in logic formulation for common 42 hurdles (e.g., memory management, process handling, or algorithm optimization).

## Tech Stack
- Frontend: Next.js (React)

- Backend: Python (FastAPI)

- AI Integration: Google Gemini API / OpenAI (via LangChain or LlamaIndex)

- Database: SQLite (for chat history and metadata)

- Evaluation: DeepEval (for monitoring faithfulness and relevancy of AI answers)

## Getting Started
Prerequisites
Node.js (v18+)

Python (3.9+)

API Key for your chosen LLM (Gemini/OpenAI)

### Installation Clone the repository

```Bash
git clone https://github.com/nirut107/42subject.ai.git
cd 42subject.ai
Backend Setup
```
```Bash
cd backend
pip install -r requirements.txt
# Create a .env file and add your API keys
python main.py
Frontend Setup
```
```Bash
cd frontend
npm install
npm run dev
```
## Usage
- Open the application in your browser (usually http://localhost:3000).



Ask specific questions like:
```
"How should I handle memory leaks in get_next_line?"

"Explain the forbidden functions for pipex."

"What are the edge cases for the printf width field?"
```

Project Status
This project is under active development. Current focus:

Improving OCR accuracy for complex PDF diagrams using Google Cloud Vision.

Implementing stricter AI guardrails to ensure academic integrity.

## Disclaimer
This tool is meant for educational purposes only. It is designed to help you understand subjects better, not to write code for you. Always adhere to the 42 Network's peer-learning principles and academic honesty policies.

### Created by nirut107