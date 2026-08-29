FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt ./
RUN pip install -r requirements.txt
COPY api/ ./api/
COPY docs/ ./docs/
ENV PORT=8080
CMD uvicorn api.app.main:app --host 0.0.0.0 --port ${PORT:-8080}
