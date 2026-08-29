import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY", "")


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Optional API-key auth: if API_KEY is unset, the API is open (local demo)."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
