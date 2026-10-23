from langchain_core.prompts import ChatPromptTemplate

ATS_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert ATS (Applicant Tracking System) recruiter and resume optimization specialist. Your job is strictly to evaluate a candidate's resume against a provided Job Description (JD)."),
    ("human", """
Analyze the candidate's resume below against the target Job Description.

---
**JOB DESCRIPTION:**
{job_description}

---
**CANDIDATE RESUME TEXT:**
{resume_text}

---
Provide your analysis strictly in the following JSON format:
{{
  "ats_score": <integer score between 0 and 100>,
  "missing_keywords": ["keyword1", "keyword2", ...],
  "mistakes_found": ["Specific mistake 1", "Specific mistake 2", ...]
}}
""")
])

RESUME_OPTIMIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an elite resume writer and career coach. Your task is to rewrite and optimize the resume so it achieves a 100/100 ATS score for the target job description, while keeping the candidate's core history honest and authentic. If resume code (like LaTeX or Markdown) is provided, you must also update the code structure to match the improved content."),
    ("human", """
Target Job Description:
{job_description}

Original Resume Text:
{resume_text}

Original Resume Code (if provided):
{resume_code}

Current Missing Keywords / Mistakes to Fix:
Missing Keywords: {missing_keywords}
Mistakes: {mistakes_found}

User/Human Feedback from previous round (if any):
{human_feedback}

Provide your response strictly in the following JSON format:
{{
  "optimized_resume_text": "Full text of the rewritten, highly optimized resume...",
  "optimized_resume_code": "Updated code of the resume (or null if no original code was provided)..."
}}
""")
])