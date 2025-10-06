from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import ORJSONResponse
from datetime import datetime
from dateutil import tz
from .security import require_api_key
from .blob_io import latest_blob, blob_text
from .parsers import csv_to_records, ndjson_to_records

app = FastAPI(title="StockSentinel Data API", default_response_class=ORJSONResponse)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/v1/stocks")
def get_stocks(date: str | None = Query(None), latest: bool = False, ticker: str | None = None, _=Depends(require_api_key)):
    prefix = "stocks/"
    if latest:
        path = latest_blob(prefix, ".csv")
    else:
        if not date: raise HTTPException(400, detail="provide date=YYYYMMDD or latest=true")
        path = f"{prefix}prices_{date}.csv"
    if not path:
        raise HTTPException(404, detail="no data found")
    recs = csv_to_records(blob_text(path))
    if ticker:
        recs = [r for r in recs if str(r.get("ticker")) == ticker]
    return {"path": path, "count": len(recs), "data": recs}

@app.get("/v1/news")
def get_news(date: str | None = Query(None), start: str | None = None, end: str | None = None, latest: bool = False, ticker: str | None = None, _=Depends(require_api_key)):
    prefix = "news/"
    paths = []
    if latest:
        lb = latest_blob(prefix, ".ndjson")
        if lb: paths = [lb]
    elif date:
        paths = [f"{prefix}news_{date}.ndjson"]
    elif start and end:
        # permite rango YYYYMMDD-YYYYMMDD
        from datetime import datetime, timedelta
        s = datetime.strptime(start, "%Y%m%d")
        e = datetime.strptime(end, "%Y%m%d")
        d = s
        while d <= e:
            paths.append(f"{prefix}news_{d.strftime('%Y%m%d')}.ndjson")
            d += timedelta(days=1)
    else:
        raise HTTPException(400, detail="provide date=YYYYMMDD, latest=true or start&end")

    data = []
    for p in paths:
        try:
            items = ndjson_to_records(blob_text(p))
            data.extend(items)
        except Exception:
            # si falta un día, lo ignoramos silenciosamente
            pass
    if ticker:
        data = [d for d in data if ticker in (d.get("tickers") or [])]
    return {"paths": paths, "count": len(data), "data": data}

@app.get("/v1/dates/{domain}")
def list_dates(domain: str, _=Depends(require_api_key)):
    # domain in {stocks, news}
    from .blob_io import list_blobs
    if domain not in {"stocks", "news"}:
        raise HTTPException(400, detail="domain must be 'stocks' or 'news'")
    names = list_blobs(f"{domain}/")
    # extrae YYYYMMDD únicos
    import re
    ds = sorted({m.group(1) for n in names for m in [re.search(r"(\d{8})", n)] if m})
    return {"domain": domain, "dates": ds}
