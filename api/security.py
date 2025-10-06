from fastapi import Header, HTTPException
import os


API_KEY = os.getenv("API_KEY")


def require_api_key(x_api_key: str | None = Header(default=None)):
if not API_KEY or x_api_key != API_KEY:
raise HTTPException(status_code=401, detail="invalid api key")