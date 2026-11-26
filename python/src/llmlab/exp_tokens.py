"""Experimento 2: quanto custa, em tokens, dizer a mesma coisa em português e em inglês?

Protocolo
---------
* Corpus paralelo aberto: WMT24++ (Google, Apache-2.0), par en -> pt_BR. São textos em inglês com
  tradução humana para o português do Brasil, de quatro domínios (literário, notícias, redes sociais,
  fala). Descarto o segmento "canário" e os marcados como fonte ruim.
* Tokenizadores: GPT-2 original (inglês), GPT-2 português (o da demo), Qwen2.5 (multilíngue) e
  o200k_base (o do GPT-4o, via tiktoken).
* Medidas: tokens totais em cada língua, razão pt/en (quanto o português custa a mais) e tokens por
  palavra. Intervalo de 95% da razão por bootstrap de segmentos (2.000 reamostragens, semente 0).
"""

from __future__ import annotations

import json
import re

import numpy as np
from llmlab.graficos import RESULTADOS, SERIE, TINTA_2, estilo, salvar

CORPUS = ("google/wmt24pp", "en-pt_BR.jsonl", "fd7405c06494bc66a57b25f55d217a72f96e60dc")
TOKENIZADORES = {
    "GPT-2 (inglês)": ("hf", "openai-community/gpt2"),
    "GPT-2 português": ("hf", "pierreguillou/gpt2-small-portuguese"),
    "Qwen2.5": ("hf", "Qwen/Qwen2.5-0.5B"),
    "o200k (GPT-4o)": ("tiktoken", "o200k_base"),
}


def carregar_corpus() -> list[tuple[str, str, str]]:
    from huggingface_hub import hf_hub_download

    repo, arquivo, revisao = CORPUS
    caminho = hf_hub_download(repo, arquivo, repo_type="dataset", revision=revisao)
    pares = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            r = json.loads(linha)
            if r.get("is_bad_source") or r.get("domain") == "canary":
                continue
            pares.append((r["domain"], r["source"], r["target"]))
    return pares


def contador(tipo: str, nome: str):
    if tipo == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(nome)
        return lambda s: len(enc.encode(s))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nome)
    return lambda s: len(tok(s, add_special_tokens=False).input_ids)


def palavras(s: str) -> int:
    return len(re.findall(r"\w+", s))


def bootstrap_razao(pt: np.ndarray, en: np.ndarray, n: int = 2000, semente: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(pt), size=(n, len(pt)))
    r = pt[idx].sum(axis=1) / en[idx].sum(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def rodar() -> dict:
    pares = carregar_corpus()
    en_txt = [p[1] for p in pares]
    pt_txt = [p[2] for p in pares]
    dominios = sorted({p[0] for p in pares})
    dom = np.array([p[0] for p in pares])
    pal_en = sum(palavras(s) for s in en_txt)
    pal_pt = sum(palavras(s) for s in pt_txt)
    res = {
        "corpus": "WMT24++ en→pt_BR (Google, Apache-2.0)",
        "segmentos": len(pares),
        "palavras": {"en": pal_en, "pt": pal_pt},
        "caracteres": {"en": sum(map(len, en_txt)), "pt": sum(map(len, pt_txt))},
        "tokenizadores": [],
    }
    exemplo = ("O banco fica perto do rio?", "Is the bank near the river?")
    for rotulo, (tipo, nome) in TOKENIZADORES.items():
        conta = contador(tipo, nome)
        en = np.array([conta(s) for s in en_txt])
        pt = np.array([conta(s) for s in pt_txt])
        lo, hi = bootstrap_razao(pt, en)
        res["tokenizadores"].append(
            {
                "nome": rotulo,
                "fonte": nome,
                "tokens_en": int(en.sum()),
                "tokens_pt": int(pt.sum()),
                "razao_pt_en": float(pt.sum() / en.sum()),
                "ic95": [lo, hi],
                "tokens_por_palavra": {"en": float(en.sum() / pal_en), "pt": float(pt.sum() / pal_pt)},
                "razao_por_dominio": {d: float(pt[dom == d].sum() / en[dom == d].sum()) for d in dominios},
                "exemplo": {"pt": conta(exemplo[0]), "en": conta(exemplo[1])},
            }
        )
    graficos(res)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "tokens.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def graficos(res: dict) -> None:
    import matplotlib.pyplot as plt

    estilo()
    ts = res["tokenizadores"]
    nomes = [t["nome"] for t in ts]
    y = np.arange(len(ts))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    h = 0.36
    en = [t["tokens_por_palavra"]["en"] for t in ts]
    pt = [t["tokens_por_palavra"]["pt"] for t in ts]
    a1.barh(y + h / 2, en, height=h - 0.04, color=SERIE[0], label="inglês")
    a1.barh(y - h / 2, pt, height=h - 0.04, color=SERIE[1], label="português")
    for yi, e, p in zip(y, en, pt):
        a1.text(e + 0.02, yi + h / 2, f"{e:.2f}", va="center", fontsize=9, color=TINTA_2)
        a1.text(p + 0.02, yi - h / 2, f"{p:.2f}", va="center", fontsize=9, color=TINTA_2)
    a1.set_yticks(y, nomes)
    a1.set_xlabel("tokens por palavra")
    a1.set_title("Tokens por palavra, mesmo conteúdo")
    a1.legend(loc="lower right", fontsize=9)
    a1.grid(axis="y", visible=False)

    r = [t["razao_pt_en"] for t in ts]
    err = np.array([[t["razao_pt_en"] - t["ic95"][0], t["ic95"][1] - t["razao_pt_en"]] for t in ts]).T
    a2.barh(y, r, height=0.5, color=SERIE[1], xerr=err, error_kw={"ecolor": TINTA_2, "lw": 1, "capsize": 3})
    a2.axvline(1, color="#b9b8b3", lw=1, ls="--")
    for yi, v in zip(y, r):
        a2.text(v + 0.05, yi, f"{v:.2f}×", va="center", fontsize=9.5, color=TINTA_2)
    a2.set_yticks(y, [""] * len(ts))
    a2.set_xlim(0, max(r) * 1.25)
    a2.set_xlabel("tokens em português ÷ tokens em inglês")
    a2.set_title("Custo do português (IC 95%)")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    salvar(fig, "tokens_pt_en.png")


if __name__ == "__main__":
    r = rodar()
    print(r["segmentos"], "segmentos", r["palavras"])
    for t in r["tokenizadores"]:
        print(f"{t['nome']:18s} en {t['tokens_en']:6d} pt {t['tokens_pt']:6d} razão {t['razao_pt_en']:.2f} "
              f"{t['ic95']}  exemplo {t['exemplo']}")
