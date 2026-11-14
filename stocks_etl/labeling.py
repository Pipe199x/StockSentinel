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


def triple_barrier_live(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        event_pos: np.ndarray, side: np.ndarray, sigma: np.ndarray,
                        k_tp: float, k_sl: float, t_max: np.ndarray, series_ended: bool = False,
                        div: np.ndarray = None) -> pd.DataFrame:
    """Triple barrera ejecutada como el bot de paper trading (stocks_etl.paper._walk).

    Take profit y stop se publican en precio desde el cierre del día de la señal; la entrada
    es la APERTURA de la sesión siguiente. Si esa apertura ya cruzó una barrera, la orden
    sale plana ('entry_gap'). Después: gap en la apertura -> sale a la apertura; toque
    intradía -> sale en la barrera; ambos el mismo día -> stop; barrera de tiempo al cierre
    de la sesión t_max contada desde la señal. Retornos medidos desde el precio de entrada.

    div: dividendo en efectivo por acción (escala de precios ajustada por splits) en la fila
    de cada fecha ex. La operación cobra (largo) o paga (corto) los dividendos con fecha ex en
    (día de entrada, día de salida]: ret = s * ((salida + D) / entrada - 1). Las barreras
    siguen sobre precios, como las órdenes reales. Devuelve también entry_px, exit_px y div.
    """
    o, h, l, c = (np.asarray(a, dtype=float) for a in (open_, high, low, close))
    n = len(c)
    m = len(event_pos)
    dv = np.zeros(n) if div is None else np.nan_to_num(np.asarray(div, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(dv)])          # sum of dv[a..b] = cum[b + 1] - cum[a]
    out_end = np.empty(m, dtype=int)
    out_ret = np.full(m, np.nan)
    out_bar = np.empty(m, dtype=object)
    out_e = np.full(m, np.nan)
    out_x = np.full(m, np.nan)
    out_d = np.zeros(m)
    for i, (p0, s, sg, tm) in enumerate(zip(event_pos, side, sigma, t_max)):
        c0 = c[p0]
        tp, sl = c0 * (1 + s * k_tp * sg), c0 * (1 - s * k_sl * sg)
        out_end[i], out_bar[i] = p0, "open"
        if p0 + 1 > n - 1:
            continue
        e = o[p0 + 1] if np.isfinite(o[p0 + 1]) else c[p0 + 1]
        out_e[i] = e
        exit_p = p0 + int(tm)
        for p in range(p0 + 1, min(exit_p, n - 1) + 1):
            px, why = None, None
            if p == p0 + 1:
                if s * (e - tp) >= 0 or s * (e - sl) <= 0:
                    px, why = e, "entry_gap"
            elif np.isfinite(o[p]):
                if s * (o[p] - tp) >= 0:
                    px, why = o[p], "tp"
                elif s * (o[p] - sl) <= 0:
                    px, why = o[p], "sl"
            if px is None:
                hit_tp = (h[p] >= tp) if s > 0 else (l[p] <= tp)
                hit_sl = (l[p] <= sl) if s > 0 else (h[p] >= sl)
                if hit_sl:
                    px, why = sl, "sl"
                elif hit_tp:
                    px, why = tp, "tp"
            if px is None and p >= exit_p:
                px, why = c[p], "t"
            if px is not None:
                d = cum[p + 1] - cum[p0 + 2] if p > p0 + 1 else 0.0   # ex-dates in (entry, exit]
                out_end[i], out_ret[i], out_bar[i] = p, s * ((px + d) / e - 1), why
                out_x[i], out_d[i] = px, d
                break
        else:
            if series_ended and exit_p > n - 1:
                last = n - 1
                d = cum[last + 1] - cum[p0 + 2] if last > p0 + 1 else 0.0
                out_end[i], out_ret[i], out_bar[i] = last, s * ((c[last] + d) / e - 1), "delist"
                out_x[i], out_d[i] = c[last], d
    return pd.DataFrame({"start_pos": event_pos, "end_pos": out_end, "ret": out_ret, "barrier": out_bar,
                         "entry_px": out_e, "exit_px": out_x, "div": out_d})


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
