# 📚 RAG with Sources + Evaluation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=langchain)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?logo=groq)

> **A smart question-answering app over your own documents. It provides accurate answers with exact source citations and includes a built-in evaluation system to measure its own performance.**

---

## 📋 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system. It allows users to ask natural language questions about their custom documents. Unlike standard chatbots, this system shows **exactly where the answer came from** and runs automated tests to ensure high accuracy.

---

## 🎯 Problem Statement
Standard search tools often give vague answers without proof. This project solves that by:
- Combining semantic retrieval with LLM generation for precise answers.
- Providing **transparent source citations** (File, Page, Snippet).
- Including an **automated evaluation pipeline** to measure retrieval and answer accuracy.

---

## 📊 Dataset
- **Custom Documents:** User-uploaded files (`.txt`, `.md`, `.pdf`) placed in the `/data` folder.
- **Evaluation Set:** A structured JSON file (`/eval/test_set.json`) containing predefined questions, expected keywords, and "unanswerable" edge cases to test the system's limits.

---

## 🛠️ Tools & Technologies

| Category | Technologies Used |
| :--- | :--- |
| **Backend & AI** | Python, LangChain, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB |
| **LLM Inference** | Groq API (Fast & Low Latency) |
| **Frontend UI** | Streamlit |
| **Evaluation** | Custom Python Evaluation Framework |

---

## 🔬 Methods

1. **Document Processing:** Loads files from `/data`, splits them into semantic chunks, and stores embeddings in ChromaDB.
2. **Semantic Retrieval:** Finds the most relevant chunks based on the user's query using vector similarity.
3. **Contextual Generation:** Groq LLM generates a precise answer using *only* the retrieved context.
4. **Automated Evaluation:** The `evaluate.py` script tests the system against `test_set.json` to calculate **Retrieval Hit Rate** and **Answer Accuracy**.

---

##  Key Insights
- **Transparency:** Users can trust the AI because every answer is backed by a visible source.
- **Speed:** Using Groq API ensures sub-second response times.
- **Reliability:** The built-in evaluation framework prevents "hallucinations" by testing the system against known facts.

---

## 🖥️ Dashboard & Output

The Streamlit UI provides a clean chat interface. 

**Example Interaction:**
```text
User: "What are the key benefits of this system?"

AI: "The key benefits include high accuracy, fast processing, and transparent sourcing."

📄 Sources:
- report.pdf (Page 2): "The system offers high accuracy..."
- notes.md (Section 1): "Fast processing is a core feature..."

✅ Retrieval Score: 0.95 | Confidence: High