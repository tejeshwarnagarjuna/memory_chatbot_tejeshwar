# Memory Chatbot

A chatbot that remembers the conversation, built with Python, LangChain, Groq (GPT-OSS 120B), Streamlit.

## Features
- Remembers earlier messages in the chat (conversation memory)
- Keeps its history short by trimming the oldest messages
- Answers in the BEPEC assistant style using few-shot examples
- Uses a free model on Groq (no paid API)
- Simple chat screen made with Streamlit, with a "Clear Conversation" button

## Files
- `backend.py`: model, prompts, memory and trimming logic
- `frontend.py`: Streamlit chat screen
- `requirements.txt`: libraries to install
- `.env`: your private settings (not uploaded to GitHub)

## How to run
1. Install Python 3.10 or newer.
2. Install the libraries:
```
   pip install -r requirements.txt
```
3. Get a free API key from https://console.groq.com
4. Create a file named `.env` in the project folder with:
```
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=openai/gpt-oss-120b
```
5. Start the app:
```
   streamlit run frontend.py
```
6. Open http://localhost:8501 in your browser.

## Try it
Type "Hi, my name is Tej", then ask "What is my name?" to see the memory work.

## Built with
Python, LangChain, Groq (GPT-OSS 120B), Streamlit