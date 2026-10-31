"""Meta-modelos deep: LSTM sobre el precio y fusión precio + noticias por cross-attention.

Nada de "fusión temprana" (pegar 768 dimensiones de texto al lado de un precio): cada
rama es un experto y la unión se hace en capas intermedias. El estado del LSTM hace de
query y le pregunta a las noticias cuál explica el movimiento; un token "sin noticias"
aprendible deja que el modelo ignore el texto cuando solo es ruido.
"""
import numpy as np
import torch
from torch import nn


class LSTMMeta(nn.Module):
    """La compuerta de olvido permite descartar patrones viejos cuando cambia el régimen."""

    def __init__(self, n_feat: int, n_static: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x):
        out, (h, _) = self.lstm(x)
        return out, h[-1]

    def forward(self, x, static, news=None, mask=None):
        _, h = self.encode(x)
        return self.head(torch.cat([h, static], dim=1)).squeeze(1)


class FusionMeta(LSTMMeta):
    def __init__(self, n_feat: int, n_static: int, hidden: int = 32, d_news: int = 768, d_model: int = 64,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__(n_feat, n_static, hidden, dropout)
        self.q = nn.Linear(hidden, d_model)
        self.kv = nn.Sequential(nn.LayerNorm(d_news), nn.Linear(d_news, d_model))
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + d_model + n_static, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x, static, news=None, mask=None):
        _, h = self.encode(x)
        q = self.q(h).unsqueeze(1)
        kv = torch.cat([self.null.expand(len(x), 1, -1), self.kv(news)], dim=1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool), ~mask], dim=1)
        ctx, _ = self.attn(q, kv, kv, key_padding_mask=pad)
        return self.head(torch.cat([h, ctx.squeeze(1), static], dim=1)).squeeze(1)


def fit_predict(model_cls, X, S, y, train_idx, test_idx, E=None, M=None, epochs: int = 8,
                lr: float = 1e-3, batch: int = 256, seed: int = 42, **kw) -> np.ndarray:
    """Entrena con el train del fold y devuelve P(acierto) para el test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(X.shape[-1], S.shape[-1], **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    t = lambda a, idx: torch.from_numpy(a[idx]) if a is not None else None
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            opt.zero_grad()
            logit = model(t(X, b), t(S, b), t(E, b), t(M, b))
            loss = loss_fn(logit, torch.from_numpy(y[b].astype(np.float32)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(test_idx), 2048):
            b = test_idx[i:i + 2048]
            out.append(torch.sigmoid(model(t(X, b), t(S, b), t(E, b), t(M, b))).numpy())
    return np.concatenate(out) if out else np.zeros(0)
