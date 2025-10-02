import glob
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "stocks_etl", "..", "ETL", "data")
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Ajusta BASE si tu output_dir es distinto

def main():
    csvs = sorted(glob.glob(os.path.join(BASE, "*_prices_*.csv")))
    preds = sorted(glob.glob(os.path.join(BASE, "predictions_*.csv")))
    if not csvs:
        print("No hay CSV de precios en", BASE)
        return
    df = pd.read_csv(csvs[-1])
    print("Precios HEAD:\n", df.head())
    print("Precios DESCRIBE:\n", df.describe())
    if preds:
        dp = pd.read_csv(preds[-1])
        print("Predicciones:\n", dp)

if __name__ == "__main__":
    main()
