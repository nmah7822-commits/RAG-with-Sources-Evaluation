# RAG with Sources + Evaluation

A question-answering app over your own documents. Every answer shows the source passages it came from, and a built-in test set measures how accurate the system is.

## What it does
- Loads `.txt`, `.md` and `.pdf` files from `data/`
- Splits them into chunks and stores embeddings in a Chroma vector store
- Retrieves the most relevant chunks and asks a Groq LLM to answer using only that context
- Shows the sources (file, page, snippet) under each answer
- Evaluates itself: retrieval hit rate and answer accuracy on `eval/test_set.json`

## Stack
Python, LangChain, Chroma, sentence-transformers (all-MiniLM-L6-v2), Groq, Streamlit

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then add your GROQ_API_KEY
streamlit run app.py
```

Run the evaluation from the terminal: `python evaluate.py`

## Use your own documents
Put files in `data/` (or upload them in the app sidebar), rebuild the index, then edit `eval/test_set.json` with questions about your files.

## How evaluation works
Each test has a question and expected keywords. Retrieval hit rate checks whether the retrieved chunks contain them. Answer accuracy checks whether the final answer contains them. Questions marked `unanswerable` must get an "I don't know" answer.

## Results
Add your own numbers here after running `python evaluate.py`.