import os
from pathlib import Path
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from pypdf import PdfReader
from docx import Document

from agent.graph import app_graph

load_dotenv()

app = FastAPI(title="ATS X-Ray Resume Optimizer", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent

# Mount static files and templates relative to BASE_DIR
if os.path.exists(BASE_DIR / "static"):
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory session store for demo graph states & thread configs
# (For production scaling, you can use LangGraph MemorySaver with Postgres/Redis)
SESSION_STORE = {}

def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.endswith(".docx"):
            doc = Document(file.file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Fallback text decode
            content = file.file.read()
            text = content.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    return text.strip()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/analyze")
async def analyze_resume(
    job_description: str = Form(...),
    resume_file: Optional[UploadFile] = File(None),
    resume_code: Optional[str] = Form(None)
):
    if not resume_file and not resume_code:
        raise HTTPException(status_code=400, detail="Please upload a resume file or paste your resume code.")

    resume_text = ""
    if resume_file:
        resume_text = extract_text_from_file(resume_file)
    
    # If resume text is sparse but code is provided, use code as primary text source
    if not resume_text and resume_code:
        resume_text = resume_code

    initial_state = {
        "resume_text": resume_text,
        "resume_code": resume_code if resume_code else "",
        "job_description": job_description,
        "ats_score": 0,
        "missing_keywords": [],
        "mistakes_found": [],
        "optimized_resume_text": "",
        "optimized_resume_code": "",
        "feedback_history": [],
        "iteration_count": 0,
        "human_satisfied": False
    }

    # Config for thread execution
    thread_config = {"configurable": {"thread_id": "session_1"}}
    SESSION_STORE["config"] = thread_config

    try:
        # Run graph until first interrupt (human feedback node)
        events = app_graph.invoke(initial_state, config=thread_config)
        
        # Get current state snapshot
        state_snapshot = app_graph.get_state(thread_config)
        
        return JSONResponse({
            "status": "success",
            "ats_score": state_snapshot.values.get("ats_score", 0),
            "missing_keywords": state_snapshot.values.get("missing_keywords", []),
            "mistakes_found": state_snapshot.values.get("mistakes_found", []),
            "optimized_resume_text": state_snapshot.values.get("optimized_resume_text", ""),
            "optimized_resume_code": state_snapshot.values.get("optimized_resume_code", ""),
            "iteration_count": state_snapshot.values.get("iteration_count", 1)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(
    satisfied: bool = Form(...),
    feedback: Optional[str] = Form("")
):
    thread_config = SESSION_STORE.get("config")
    if not thread_config:
        raise HTTPException(status_code=400, detail="No active session found. Please analyze a resume first.")

    try:
        # Resume the graph with human feedback payload
        human_payload = {
            "satisfied": satisfied,
            "feedback": feedback
        }
        
        events = app_graph.invoke(human_payload, config=thread_config)
        state_snapshot = app_graph.get_state(thread_config)
        
        # Check if graph has finished or is paused at another interrupt
        is_completed = len(state_snapshot.next) == 0

        return JSONResponse({
            "status": "success",
            "completed": is_completed,
            "ats_score": state_snapshot.values.get("ats_score", 100),
            "missing_keywords": state_snapshot.values.get("missing_keywords", []),
            "mistakes_found": state_snapshot.values.get("mistakes_found", []),
            "optimized_resume_text": state_snapshot.values.get("optimized_resume_text", ""),
            "optimized_resume_code": state_snapshot.values.get("optimized_resume_code", ""),
            "iteration_count": state_snapshot.values.get("iteration_count", 1)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback loop error: {str(e)}")