import pandas as pd
import io, json
import ndjson as ndj

def csv_to_records(csv_text: str) -> list[dict]:
    df = pd.read_csv(io.StringIO(csv_text))
    # tip: convertir NaN a None para JSON limpio
    return json.loads(df.to_json(orient="records"))

def ndjson_to_records(ndjson_text: str) -> list[dict]:
    return list(ndj.loads(ndjson_text))
