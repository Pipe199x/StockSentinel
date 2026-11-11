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


def triple_barrier_ohlc(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        event_pos: np.ndarray, side: np.ndarray, sigma: np.ndarray,
                        k_tp: float, k_sl: float, t_max: np.ndarray,
                        resolve=None, series_ended: bool = False) -> pd.DataFrame:
    """Triple barrera con toques intradía (máximos/mínimos), no solo cierres.

    Entrada al cierre del día del evento. Cada día siguiente:
      - si abre más allá de una barrera (gap), sale a la apertura (peor o mejor que la barrera);
      - si el máximo/mínimo toca una barrera, sale exactamente en ella;
      - si toca las dos el mismo día, ``resolve(pos, tp_price, sl_price, side)`` decide con
        barras de minuto ('tp' | 'sl' | None). Sin resolver, se asume el stop (conservador).
    Si la serie termina (deslistado) con la operación abierta, sale al último cierre
    (barrier='delist') en lugar de descartar el evento: descartarlo esconde las pérdidas.
    Columna extra ``ambiguous``: el día de salida tocó ambas barreras.
    """
    o, h, l, c = (np.asarray(a, dtype=float) for a in (open_, high, low, close))
    n = len(c)
    m = len(event_pos)
    out_end = np.empty(m, dtype=int)
    out_ret = np.full(m, np.nan)
    out_bar = np.empty(m, dtype=object)
    out_amb = np.zeros(m, dtype=bool)
    for i, (p0, s, sg, tm) in enumerate(zip(event_pos, side, sigma, t_max)):
        c0 = c[p0]
        up, dn = k_tp * sg, k_sl * sg
        tp_px = c0 * (1 + s * up)
        sl_px = c0 * (1 - s * dn)
        last = min(p0 + int(tm), n - 1)
        out_end[i], out_bar[i] = p0, "open"
        for p in range(p0 + 1, last + 1):
            r_open = s * (o[p] / c0 - 1) if np.isfinite(o[p]) else np.nan
            if np.isfinite(r_open) and r_open >= up:
                out_end[i], out_ret[i], out_bar[i] = p, r_open, "tp"
                break
            if np.isfinite(r_open) and r_open <= -dn:
                out_end[i], out_ret[i], out_bar[i] = p, r_open, "sl"
                break
            best, worst = (h[p], l[p]) if s > 0 else (l[p], h[p])
            hit_tp = s * (best / c0 - 1) >= up
            hit_sl = s * (worst / c0 - 1) <= -dn
            if hit_tp and hit_sl:
                out_amb[i] = True
                first = resolve(p, tp_px, sl_px, int(s)) if resolve is not None else None
                hit_tp, hit_sl = (first == "tp"), (first != "tp")
            if hit_tp:
                out_end[i], out_ret[i], out_bar[i] = p, up, "tp"
                break
            if hit_sl:
                out_end[i], out_ret[i], out_bar[i] = p, -dn, "sl"
                break
        else:
            if last > p0 and p0 + int(tm) <= n - 1:
                out_end[i], out_ret[i], out_bar[i] = last, s * (c[last] / c0 - 1), "t"
            elif last > p0 and series_ended:
                out_end[i], out_ret[i], out_bar[i] = last, s * (c[last] / c0 - 1), "delist"
    return pd.DataFrame({"start_pos": event_pos, "end_pos": out_end, "ret": out_ret,
                         "barrier": out_bar, "ambiguous": out_amb})
