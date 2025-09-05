# config.py
from __future__ import annotations
from dotenv import load_dotenv
import os

load_dotenv()  # carga variables desde .env si existe

SUPABASE_URL: str | None = os.getenv("SUPABASE_API_URL")
SUPABASE_KEY: str | None = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
NEWS_API_KEY: str | None = os.getenv("NEWS_API_KEY")

# Permite definir la lista por .env, si no, usa un fallback seguro
_stocks_raw = os.getenv("STOCKS_TO_FETCH", "AMZN,MSFT,GOOGL")
STOCKS_TO_FETCH = [s.strip().upper() for s in _stocks_raw.split(",") if s.strip()]

# Validación mínima (opcional pero recomendable)
_missing = [name for name, val in {
    "SUPABASE_API_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_KEY,
    "NEWS_API_KEY": NEWS_API_KEY,
}.items() if not val]

if _missing:
    msg = (
        "Missing required environment variables: "
        + ", ".join(_missing)
        + ". Create a .env file or set them in the environment."
    )
    # Puedes cambiar a warnings si prefieres no romper la app:
    raise RuntimeError(msg)
