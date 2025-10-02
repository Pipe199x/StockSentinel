import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="data/news_YYYYMMDD.ndjson")
    args = ap.parse_args()
    df = pd.read_json(args.file, lines=True, dtype=False)

    print("=== HEAD ===")
    print(df[["published_at","source","title","language","sentiment"]].head(5))

    print("\n=== Conteos por idioma ===")
    print(df["language"].apply(lambda x: (x or {}).get("iso") if isinstance(x, dict) else None).value_counts(dropna=False))

    print("\n=== Conteos por sentimiento (label) ===")
    print(df["sentiment"].apply(lambda x: (x or {}).get("label") if isinstance(x, dict) else None).value_counts(dropna=False))

    def get_kp(x): return x if isinstance(x, list) else []
    kp = df["key_phrases"].apply(get_kp).explode().dropna()
    print("\n=== Top 20 key phrases ===")
    print(kp.value_counts().head(20))

if __name__ == "__main__":
    main()
