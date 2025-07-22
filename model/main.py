from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader # type: ignore
import io
import time
import json

app = FastAPI()

# CORS (optional kalau diakses dari frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = SentenceTransformer("all-mpnet-base-v2")


@app.post("/recommend-jobs")
async def recommend_jobs(
    resume: UploadFile = File(...),
    job_descriptions: str = Form(...)
):

    start_time = time.time()

    # 1. Extract PDF text
    try:
        pdf_bytes = await resume.read()
        text = "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf_bytes)).pages])
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read PDF: {str(e)}"})

    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "No text extracted from PDF"})

    # 2. Load jobs
    try:
        jobs = json.loads(job_descriptions)
        if not isinstance(jobs, list):
            raise ValueError()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid job_descriptions: {e}"})

    # 3. Vectorize
    resume_emb = model.encode(text, convert_to_tensor=True)
    jobs_emb = model.encode(jobs, convert_to_tensor=True)

    # 4. Similarity + Filter (min_score tetap)
    scores = util.pytorch_cos_sim(resume_emb, jobs_emb)[0].tolist()
    min_score = 0.25  # <- bisa kamu ganti sesukamu
    results = [
        {"index": i, "job_description": jobs[i], "score": round(s, 4)}
        for i, s in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if s >= min_score
    ]

    return {
        "resume_text": text[:500] + "...",
        "results": results,
        "processed_in_seconds": round(time.time() - start_time, 2)
    }
