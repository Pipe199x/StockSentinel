FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt ./
RUN pip install -r requirements.txt
COPY api/ ./
ENV PORT=8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]