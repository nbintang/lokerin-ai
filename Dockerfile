# Gunakan image dasar python
FROM python:3.10-slim

# Tambahkan user non-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Atur direktori kerja
WORKDIR /model

# Salin dan install dependencies
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Salin seluruh file project (kecuali yang di .dockerignore/.gitignore)
COPY --chown=user . .

# Expose port
EXPOSE 7860

# Jalankan FastAPI dengan Uvicorn dari app/main.py
CMD ["uvicorn", "model.main:app", "--host", "0.0.0.0", "--port", "7860"]
