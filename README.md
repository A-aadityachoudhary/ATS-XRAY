# ⚡ ATS X-Ray | AI-Powered Resume Optimizer & Diagnostic Agent

An intelligent, agentic ATS (Applicant Tracking System) resume analyzer and rewriter built with **LangChain**, **LangGraph**, **Groq LLM**, **FastAPI**, and **Tailwind CSS**. 

ATS X-Ray evaluates a candidate's resume against a target job description, calculates a precise ATS match score out of 100, pinpoints formatting flaws and missing keywords, and automatically re-writes the resume (including code snippets like LaTeX/Markdown) targeting a **100/100 match score**. It features a **Human-in-the-Loop** execution graph that iterates until the user is satisfied.

---

## ✨ Features

- **🎯 ATS Match Scorecard**: Evaluates your resume against a target Job Description and calculates an alignment score (0–100).
- **🔍 Deep Flaw & Keyword Detection**: Extracts missing hard/soft skills and identifies specific structural, formatting, or content mistakes.
- **✨ Automated Optimization & Code Update**: Rewrites your resume text and optional source code (LaTeX, Markdown, HTML, etc.) to optimize keyword density and impact.
- **🔄 Human-in-the-Loop Feedback Loop**: Built with LangGraph state interrupts—request custom changes and iteratively refine the output until satisfied.
- **⚡ Fast Python Workspace**: Managed with **`uv`** for lightning-fast environment setup and dependency installation.
- **🎨 Responsive `#FAFFC4` Theme**: Modern, glassmorphic UI styled with Tailwind CSS and interactive state using Alpine.js.
- **🚀 Powered by Groq**: Free, low-latency LLM execution using Groq's high-speed inference engine.

---

## 🛠️ Tech Stack

- **Backend / Agent**: Python 3.10+, FastAPI, LangGraph, LangChain, `langchain-groq`
- **LLM Engine**: Groq API (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, or `mixtral-8x7b-32768`)
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv)
- **Document Parsers**: `pypdf`, `python-docx`
- **Frontend**: HTML5, Tailwind CSS, Alpine.js, Jinja2 Templates

---

## 📁 Project Structure

```text
ATS-XRAY/
├── .env                  # Groq API environment variables
├── pyproject.toml        # Dependencies and project config
├── uv.lock               # Lockfile for uv package manager
├── main.py               # FastAPI server & route handlers
├── agent/
│   ├── __init__.py
│   ├── state.py          # LangGraph ResumeState schema
│   ├── graph.py          # LangGraph workflow nodes & checkpointer
│   └── prompts.py        # System prompts for scoring & rewriting
├── static/               # Static assets
├── templates/
│   └── index.html        # Responsive frontend template
└── README.md

---

## Command To Run project
- uv run uvicorn main:app --reload --port 8000
