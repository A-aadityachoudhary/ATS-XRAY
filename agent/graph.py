import os
import json
from typing import Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from agent.state import ResumeState
from agent.prompts import ATS_EVAL_PROMPT, RESUME_OPTIMIZE_PROMPT

load_dotenv()

# Initialize Groq LLM (using llama-3.3-70b-versatile, which is fast and free-tier friendly on Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    api_key=os.getenv("GROQ_API_KEY")
)

def evaluate_resume(state: ResumeState):
    """Node 1: Evaluates ATS score, missing keywords, and mistakes."""
    prompt = ATS_EVAL_PROMPT.format(
        job_description=state["job_description"],
        resume_text=state["resume_text"]
    )
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Clean up potential markdown formatting in JSON response
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        data = json.loads(content)
        ats_score = data.get("ats_score", 50)
        missing_keywords = data.get("missing_keywords", [])
        mistakes_found = data.get("mistakes_found", [])
    except Exception:
        ats_score = 50
        missing_keywords = ["Unable to parse keywords"]
        mistakes_found = ["Error parsing LLM evaluation output."]
        
    return {
        "ats_score": ats_score,
        "missing_keywords": missing_keywords,
        "mistakes_found": mistakes_found,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def optimize_resume(state: ResumeState):
    """Node 2: Rewrites the resume text and code to achieve a 100/100 ATS score."""
    # Gather feedback history if any
    history = state.get("feedback_history", [])
    latest_feedback = history[-1] if history else "None provided yet. Make it a 100/100 ATS-optimized resume."
    
    prompt = RESUME_OPTIMIZE_PROMPT.format(
        job_description=state["job_description"],
        resume_text=state["resume_text"],
        resume_code=state.get("resume_code", "None provided"),
        missing_keywords=", ".join(state.get("missing_keywords", [])),
        mistakes_found=", ".join(state.get("mistakes_found", [])),
        human_feedback=latest_feedback
    )
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        data = json.loads(content)
        optimized_text = data.get("optimized_resume_text", state["resume_text"])
        optimized_code = data.get("optimized_resume_code", state.get("resume_code"))
    except Exception:
        optimized_text = state["resume_text"]
        optimized_code = state.get("resume_code")
        
    return {
        "optimized_resume_text": optimized_text,
        "optimized_resume_code": optimized_code
    }

def human_feedback_node(state: ResumeState):
    """Node 3: Human-in-the-loop interruption point."""
    # LangGraph interrupt pauses graph execution and waits for human input
    human_response = interrupt({
        "question": "Are you satisfied with this optimized resume version, or would you like further changes?",
        "ats_score": state["ats_score"],
        "optimized_resume_text": state["optimized_resume_text"],
        "optimized_resume_code": state.get("optimized_resume_code")
    })
    
    # human_response can be a dict or string containing feedback / satisfaction status
    if isinstance(human_response, dict):
        satisfied = human_response.get("satisfied", False)
        feedback = human_response.get("feedback", "")
    else:
        satisfied = True if str(human_response).lower() in ["yes", "satisfy", "done", "ok"] else False
        feedback = str(human_response)
        
    history = state.get("feedback_history", [])
    if feedback:
        history.append(feedback)
        
    return {
        "human_satisfied": satisfied,
        "feedback_history": history
    }

def should_continue(state: ResumeState) -> Literal["optimize", "__end__"]:
    """Conditional routing based on whether the human is satisfied."""
    if state.get("human_satisfied", False) or state.get("ats_score", 0) >= 98:
        return "__end__"
    return "optimize"

# Build the Graph
workflow = StateGraph(ResumeState)

workflow.add_node("evaluate", evaluate_resume)
workflow.add_node("optimize", optimize_resume)
workflow.add_node("human_feedback", human_feedback_node)

workflow.set_entry_point("evaluate")
workflow.add_edge("evaluate", "optimize")
workflow.add_edge("optimize", "human_feedback")
workflow.add_conditional_edges(
    "human_feedback",
    should_continue,
    {
        "optimize": "evaluate", # Loop back to re-evaluate after changes
        "__end__": END
    }
)

app_graph = workflow.compile()