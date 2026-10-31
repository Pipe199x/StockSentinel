"""Método de la triple barrera con volatilidad dinámica.

Para cada evento (día en que la señal primaria dice comprar/vender):
  - take profit:  +k_tp · σ   (σ = volatilidad GARCH escalada al horizonte)
  - stop loss:    -k_sl · σ
  - barrera de tiempo: t_max sesiones (vida media OU), libera el capital si no toca ninguna.
Si la volatilidad se duplica, las barreras se ensanchan en la misma proporción.
"""
import numpy as np
import pandas as pd


def triple_barrier(close: np.ndarray, event_pos: np.ndarray, side: np.ndarray, sigma: np.ndarray,
                   k_tp: float, k_sl: float, t_max: np.ndarray) -> pd.DataFrame:
    """Etiqueta eventos sobre una serie de cierres (posiciones enteras).

    Devuelve por evento: end_pos (posición del toque), ret (retorno del lado operado,
    simple), barrier ('tp' | 'sl' | 't'). Los eventos sin datos suficientes hacia
    adelante usan la última barra disponible y quedan marcados barrier='open'.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    out_end = np.empty(len(event_pos), dtype=int)
    out_ret = np.empty(len(event_pos), dtype=float)
    out_bar = np.empty(len(event_pos), dtype=object)
    for i, (p0, s, sg, tm) in enumerate(zip(event_pos, side, sigma, t_max)):
        last = min(p0 + int(tm), n - 1)
        path = close[p0 + 1:last + 1] / close[p0] - 1.0
        path = path * s
        hit_tp = np.flatnonzero(path >= k_tp * sg)
        hit_sl = np.flatnonzero(path <= -k_sl * sg)
        first_tp = hit_tp[0] if len(hit_tp) else np.inf
        first_sl = hit_sl[0] if len(hit_sl) else np.inf
        if first_tp == np.inf and first_sl == np.inf:
            j = len(path) - 1
            out_bar[i] = "t" if p0 + int(tm) <= n - 1 else "open"
        elif first_tp <= first_sl:
            j = int(first_tp)
            out_bar[i] = "tp"
        else:
            j = int(first_sl)
            out_bar[i] = "sl"
        if j < 0:  # evento en la última barra: aún sin camino
            out_end[i], out_ret[i], out_bar[i] = p0, np.nan, "open"
            continue
        out_end[i] = p0 + 1 + j
        out_ret[i] = path[j]
    return pd.DataFrame({"start_pos": event_pos, "end_pos": out_end, "ret": out_ret, "barrier": out_bar})
