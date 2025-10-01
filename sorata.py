# prueba rápida para ver si tu CSV tiene dividendos > 0 por ticker
import pandas as pd
df = pd.read_csv("data/AMZN_prices_20251001.csv")  # cambia nombre al último archivo MSFT
print(df[df["dividends"] > 0][["date","dividends"]].head(10))
