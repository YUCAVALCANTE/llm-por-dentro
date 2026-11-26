"""Logits -> probabilidades -> próximo token.

É a mesma lógica de `web/src/amostragem.js` (a demo no navegador), linha por linha:

1. divide os logits pela temperatura (temperatura 0 = escolha gulosa, sempre o mais provável);
2. softmax;
3. ordena do mais para o menos provável (empate: menor ID primeiro);
4. top-k: fica só com os k primeiros (k = 0 desliga);
5. top-p: fica com o menor prefixo cuja probabilidade acumulada chega a p (p = 1 desliga);
6. renormaliza e sorteia com um gerador de números com semente (mulberry32), o mesmo do JS,
   para que a mesma semente dê a mesma escolha nos dois lados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & _M32


class Mulberry32:
    """Gerador pseudoaleatório de 32 bits; idêntico ao `mulberry32` de `web/src/amostragem.js`."""

    def __init__(self, semente: int):
        self.a = semente & _M32

    def __call__(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & _M32
        t = self.a
        t = _imul(t ^ (t >> 15), t | 1)
        t = ((t + _imul(t ^ (t >> 7), t | 61)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296


def softmax(logits: np.ndarray, temperatura: float = 1.0) -> np.ndarray:
    """Softmax numericamente estável. Temperatura <= 0 vira um vetor one-hot no argmax."""
    z = np.asarray(logits, dtype=np.float64)
    if temperatura <= 0:
        p = np.zeros_like(z)
        p[int(np.argmax(z))] = 1.0
        return p
    z = z / temperatura
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Distribuicao:
    probs: np.ndarray  # probabilidades depois do softmax (antes dos cortes)
    ordem: np.ndarray  # índices em ordem decrescente de probabilidade
    mantidos: np.ndarray  # índices que sobreviveram a top-k e top-p, na mesma ordem
    finais: np.ndarray  # probabilidades renormalizadas dos `mantidos`


def distribuicao(logits: np.ndarray, temperatura: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> Distribuicao:
    p = softmax(logits, temperatura)
    ids = np.arange(len(p))
    ordem = np.lexsort((ids, -p))  # prob. decrescente; empate -> menor ID
    n = len(ordem)
    if top_k and top_k > 0:
        n = min(n, int(top_k))
    if top_p < 1.0:
        acum = np.cumsum(p[ordem[:n]])
        n = min(n, int(np.searchsorted(acum, top_p - 1e-12)) + 1)
    n = max(n, 1)
    mantidos = ordem[:n]
    finais = p[mantidos] / p[mantidos].sum()
    return Distribuicao(p, ordem, mantidos, finais)


def sortear(d: Distribuicao, rng: Mulberry32) -> int:
    u = rng()
    acum = 0.0
    for idx, pr in zip(d.mantidos, d.finais):
        acum += pr
        if u < acum:
            return int(idx)
    return int(d.mantidos[-1])


def proximo_token(logits, temperatura=1.0, top_k=0, top_p=1.0, rng: Mulberry32 | None = None) -> int:
    d = distribuicao(logits, temperatura, top_k, top_p)
    if temperatura <= 0 or rng is None:
        return int(d.mantidos[0])
    return sortear(d, rng)
