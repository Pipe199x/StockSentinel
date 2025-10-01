import pandas as pd
from typing import List, Tuple

# Valida DF de precios (post-transform)

def validate_stocks_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    issues = []
    required = ["date", "ticker", "close"]
    for c in required:
        if c not in df.columns:
            issues.append(f"Falta columna requerida: {c}")
    if df.empty:
        issues.append("DataFrame vacío")
    if "close" in df.columns and (df["close"] <= 0).any():
        issues.append("Valores de close <= 0 detectados")
    # Duplicados por (ticker,date)
    if {"ticker", "date"}.issubset(df.columns):
        dups = df.duplicated(subset=["ticker", "date"]).sum()
        if dups > 0:
            issues.append(f"Duplicados (ticker,date): {dups}")
    return (len(issues) == 0, issues)

# Valida DF de predicciones (retornos)

def validate_stock_preds_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    issues = []
    for c in ["ticker", "as_of", "pred_ret_t1"]:
        if c not in df.columns:
            issues.append(f"Falta columna requerida: {c}")
    if df.empty:
        issues.append("Predicciones vacías")
    if "pred_ret_t1" in df.columns and df["pred_ret_t1"].isna().any():
        issues.append("pred_ret_t1 contiene NaN")
    return (len(issues) == 0, issues)
