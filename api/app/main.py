"""StockSentinel Data API — local-runnable serving layer.

Production delivery is static (GitHub Pages serves docs/data/ directly); this
FastAPI app exposes the same data as a conventional REST API for local use:

    uvicorn api.app.main:app --reload   # from the repo root
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from .security import require_api_key
from .local_io import read_json, read_text
from .parsers import csv_to_records, ndjson_to_records

app = FastAPI(title="StockSentinel Data API", default_response_class=ORJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/manifest")
def get_manifest(_=Depends(require_api_key)):
    data = read_json("manifest.json")
    if data is None:
        raise HTTPException(404, detail="manifest not found; run the ETLs first")
    return data


@app.get("/v1/stocks")
def get_stocks(ticker: str | None = None, _=Depends(require_api_key)):
    data = read_json("prices_latest.json")
    if data is None:
        raise HTTPException(404, detail="no stock data; run stocks_etl first")
    if ticker:
        series = data.get("series", {})
        if ticker not in series:
            raise HTTPException(404, detail=f"unknown ticker {ticker}")
        data = {**data, "series": {ticker: series[ticker]}}
    return data


@app.get("/v1/stocks/history")
def get_stocks_history(ticker: str | None = None, _=Depends(require_api_key)):
    text = read_text("history/prices_history.csv")
    if text is None:
        raise HTTPException(404, detail="no price history; run stocks_etl first")
    recs = csv_to_records(text)
    if ticker:
        recs = [r for r in recs if str(r.get("ticker")) == ticker]
    return {"count": len(recs), "data": recs}


@app.get("/v1/predictions")
def get_predictions(ticker: str | None = None, _=Depends(require_api_key)):
    data = read_json("predictions_latest.json")
    if data is None:
        raise HTTPException(404, detail="no predictions; run stocks_etl first")
    if ticker:
        data = [r for r in data if str(r.get("ticker")) == ticker]
    return {"count": len(data), "data": data}


@app.get("/v1/news")
def get_news(ticker: str | None = None, history: bool = False, _=Depends(require_api_key)):
    if history:
        text = read_text("history/news_history.ndjson")
        if text is None:
            raise HTTPException(404, detail="no news history; run news_etl first")
        articles = ndjson_to_records(text)
    else:
        payload = read_json("news_latest.json")
        if payload is None:
            raise HTTPException(404, detail="no news data; run news_etl first")
        articles = payload.get("articles", [])
    if ticker:
        articles = [a for a in articles if ticker in (a.get("tickers") or [])]
    return {"count": len(articles), "data": articles}
