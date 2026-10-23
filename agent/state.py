from typing import Annotated, List, Optional, TypedDict
import operator

class ResumeState(TypedDict):
    resume_text: str                # Extracted text from uploaded PDF/DOCX
    resume_code: Optional[str]      # Optional code snippet (LaTeX, Markdown, HTML, etc.)
    job_description: str            # Target job description
    ats_score: int                  # Score out of 100
    missing_keywords: List[str]     # Key skills/keywords missing
    mistakes_found: List[str]       # Detailed breakdown of mistakes/formatting issues
    optimized_resume_text: str      # Rewritten resume text targeting 100/100
    optimized_resume_code: Optional[str] # Rewritten code snippet if code was provided
    feedback_history: List[str]     # History of human feedback across iterations
    iteration_count: int            # Track loop iterations
    human_satisfied: bool           # Flag for human-in-the-loop loop continuation