from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text
import io
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional
import gc
import torch
import httpx
import asyncio

app = FastAPI(title="Job Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OPTIMAL MODEL untuk Hugging Face Spaces
MODEL_NAME = "all-MiniLM-L6-v2"  # 90MB, fast & accurate

# Initialize model with optimizations
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Optimize for CPU (HF Spaces typically CPU-only)
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")
    # CPU optimizations
    torch.set_num_threads(2)  # Limit threads for HF Spaces

model = model.to(device)

# Cache dengan memory limit untuk HF Spaces
MAX_CACHE_SIZE = 5  # Limit cache untuk memory
job_embeddings_cache: Dict[str, torch.Tensor] = {}
jobs_cache: Dict[str, List] = {}

def cleanup_cache():
    """Clean old cache entries if too many"""
    if len(job_embeddings_cache) > MAX_CACHE_SIZE:
        # Remove oldest entries
        keys_to_remove = list(job_embeddings_cache.keys())[:-MAX_CACHE_SIZE]
        for key in keys_to_remove:
            del job_embeddings_cache[key]
            if key in jobs_cache:
                del jobs_cache[key]
        gc.collect()

def get_jobs_hash(jobs_data: str) -> str:
    """Generate hash for job data"""
    return hashlib.md5(jobs_data.encode()).hexdigest()[:16]  # Shorter hash

async def download_pdf_from_url(url: str) -> bytes:
    """Download PDF from URL with timeout and size limits"""
    try:
        timeout = httpx.Timeout(30.0)  # 30 second timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'application/pdf' not in content_type:
                # Some servers don't set proper content-type, check file extension or content
                if not url.lower().endswith('.pdf'):
                    # Try to detect PDF by magic bytes
                    content = response.content
                    if not content.startswith(b'%PDF'):
                        raise HTTPException(
                            status_code=400, 
                            detail="URL does not point to a PDF file"
                        )
            
            # Check file size (10MB limit)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail="PDF file too large (max 10MB)"
                )
            
            pdf_bytes = response.content
            
            # Double check size after download
            if len(pdf_bytes) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail="PDF file too large (max 10MB)"
                )
                
            return pdf_bytes
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to download PDF: HTTP {e.response.status_code}"
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=400, 
            detail="Timeout while downloading PDF from URL"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to download PDF: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "Job Recommendation API",
        "model": MODEL_NAME,
        "device": str(device),
        "status": "ready",
        "supported_inputs": ["PDF file upload", "PDF URL"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cached_datasets": len(job_embeddings_cache)
    }

@app.post("/recommend-jobs")
async def recommend_jobs(
    resume: Optional[UploadFile] = File(None),
    resume_url: Optional[str] = Form(None),
    job_data: str = Form(...),
    min_score: float = Form(0.35),  # Configurable threshold
    max_results: int = Form(50)     # Limit results
):
    start_time = time.time()
    
    # Validate input: either file upload or URL must be provided
    if not resume and not resume_url:
        return JSONResponse(
            status_code=400,
            content={"error": "Either resume file or resume_url must be provided"}
        )
    
    if resume and resume_url:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide either resume file or resume_url, not both"}
        )

    # 1. Get PDF bytes (either from upload or URL)
    filename = "unknown"
    try:
        if resume_url:
            # Download from URL
            if not resume_url.startswith(('http://', 'https://')):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid URL format. Must start with http:// or https://"}
                )
            
            pdf_bytes = await download_pdf_from_url(resume_url)
            filename = resume_url.split('/')[-1] or "downloaded.pdf"
            
        else:
            # Upload file - resume is guaranteed to be not None here due to validation above
            if resume is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Resume file is required when resume_url is not provided"}
                )
            
            if not resume.filename or not resume.filename.lower().endswith('.pdf'):
                return JSONResponse(
                    status_code=400, 
                    content={"error": "Only PDF files are supported"}
                )
            
            pdf_bytes = await resume.read()
            if len(pdf_bytes) > 10 * 1024 * 1024:  # 10MB limit
                return JSONResponse(
                    status_code=400, 
                    content={"error": "PDF file too large (max 10MB)"}
                )
            filename = resume.filename
            
    except HTTPException:
        raise  # Re-raise HTTPException as is
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Failed to get PDF: {str(e)}"}
        )

    # 2. Extract PDF text
    try:
        text = extract_text(io.BytesIO(pdf_bytes))
        
        # Clean up
        del pdf_bytes
        gc.collect()
        
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Failed to read PDF: {str(e)}"}
        )

    if not text.strip():
        return JSONResponse(
            status_code=400, 
            content={"error": "No text extracted from PDF"}
        )

    # Limit text length untuk memory efficiency
    if len(text) > 10000:
        text = text[:10000] + "..."

    # 3. Parse jobs with validation
    try:
        jobs = json.loads(job_data)
        if not isinstance(jobs, list):
            raise ValueError("Job data must be a list")
            
        if len(jobs) > 1000:  # Limit untuk HF Spaces
            return JSONResponse(
                status_code=400,
                content={"error": "Maximum 1000 jobs allowed"}
            )
            
        # Validate job structure
        valid_jobs = []
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue
            if all(key in job for key in ["id", "title", "description", "location"]):
                # Limit description length
                if len(job["description"]) > 2000:
                    job["description"] = job["description"][:2000] + "..."
                valid_jobs.append(job)
                
        if not valid_jobs:
            raise ValueError("No valid jobs found")
            
        jobs = valid_jobs
        
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Invalid job_data: {e}"}
        )

    # 4. Get or compute job embeddings
    jobs_hash = get_jobs_hash(job_data)
    
    if jobs_hash in job_embeddings_cache:
        jobs_emb = job_embeddings_cache[jobs_hash]
        cached_jobs = jobs_cache[jobs_hash]
        cache_hit = True
    else:
        # Encode jobs in batches
        job_descs = [f"{job['title']} {job['location']}. {job['description']}"  for job in jobs]
        try:
            jobs_emb = model.encode(
                job_descs,
                convert_to_tensor=True,
                batch_size=16,  # Smaller batch for memory
                show_progress_bar=False,
                device=device
            )
            
            # Cache with cleanup
            cleanup_cache()
            job_embeddings_cache[jobs_hash] = jobs_emb
            jobs_cache[jobs_hash] = jobs
            cache_hit = False
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to encode jobs: {str(e)}"}
            )

    # 5. Encode resume
    try:
        resume_emb = model.encode(
            text, 
            convert_to_tensor=True,
            device=device
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to encode resume: {str(e)}"}
        )

    # 6. Calculate similarities
    try:
        scores = util.pytorch_cos_sim(resume_emb, jobs_emb)[0]
        
        # Convert to numpy for faster processing
        scores_np = scores.cpu().numpy()
        
        # Filter and get top results efficiently
        valid_indices = np.where(scores_np >= min_score)[0]
        
        if len(valid_indices) == 0:
            ranked = []
        else:
            # Get top indices sorted by score
            top_indices = valid_indices[np.argsort(scores_np[valid_indices])[::-1]]
            
            # Limit results
            top_indices = top_indices[:max_results]
            
            ranked = [
                {
                    "id": jobs[i]["id"],
                    "title": jobs[i]["title"],
                    "score": round(float(scores_np[i]), 4)
                }
                for i in top_indices
            ]
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to calculate similarities: {str(e)}"}
        )

    # Cleanup
    del resume_emb
    if not cache_hit:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    processing_time = round(time.time() - start_time, 2)

    return {
        "resume_preview": text[:300] + "..." if len(text) > 300 else text,
        "results": ranked,
        "metadata": {
            "input_type": "url" if resume_url else "file_upload",
            "source": resume_url if resume_url else filename,
            "total_jobs_processed": len(jobs),
            "matching_jobs": len(ranked),
            "min_score_threshold": min_score,
            "processing_time_seconds": processing_time,
            "cache_hit": cache_hit,
            "model_used": MODEL_NAME
        }
    }

# New endpoint specifically for URL-based recommendations
@app.post("/recommend-jobs-by-url")
async def recommend_jobs_by_url(
    resume_url: str = Form(...),
    job_data: str = Form(...),
    min_score: float = Form(0.35),
    max_results: int = Form(50)
):
    """Endpoint khusus untuk rekomendasi berdasarkan URL PDF"""
    return await recommend_jobs(
        resume=None,
        resume_url=resume_url,
        job_data=job_data,
        min_score=min_score,
        max_results=max_results
    )

@app.post("/precompute-jobs")
async def precompute_jobs(job_data: str = Form(...)):
    """Pre-compute job embeddings untuk performance"""
    try:
        jobs = json.loads(job_data)
        if len(jobs) > 1000:
            return JSONResponse(
                status_code=400,
                content={"error": "Maximum 1000 jobs for precompute"}
            )
            
        jobs_hash = get_jobs_hash(job_data)
        
        if jobs_hash in job_embeddings_cache:
            return {
                "message": "Jobs already cached",
                "hash": jobs_hash,
                "jobs_count": len(jobs)
            }
        
        start_time = time.time()
        job_descs = [f"{job['title']}. {job['description']}" for job in jobs]
        
        jobs_emb = model.encode(
            job_descs,
            convert_to_tensor=True,
            batch_size=16,
            show_progress_bar=False
        )
        
        cleanup_cache()
        job_embeddings_cache[jobs_hash] = jobs_emb
        jobs_cache[jobs_hash] = jobs
        
        return {
            "message": "Jobs precomputed successfully",
            "jobs_count": len(jobs),
            "hash": jobs_hash,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global job_embeddings_cache, jobs_cache
    job_embeddings_cache.clear()
    jobs_cache.clear()
    gc.collect()
