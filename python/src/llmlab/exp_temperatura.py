"""Experimento 3: o que a temperatura faz com a diversidade e com a repetição?

Protocolo
---------
* 3 prompts em português, 7 temperaturas (0 = guloso, 0,3 a 1,5), 20 sementes por combinação,
  40 tokens novos, sem top-k nem top-p (para isolar a temperatura). O sorteio é o mesmo da demo
  (`llmlab.amostragem`, gerador mulberry32): semente s gera a amostra s.
* Medidas, por prompt e temperatura:
  - saídas distintas: quantas das 20 continuações são diferentes;
  - distinct-2: bigramas de tokens únicos ÷ bigramas totais, juntando as 20 amostras (diversidade entre amostras);
  - repetição interna: fração de trigramas repetidos dentro de cada amostra (média nas 20);
  - surpresa: -log p (em nats) de cada token escolhido segundo o próprio modelo a T = 1.
    Quanto maior, mais "improvável" o texto é para o modelo.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from llmlab.amostragem import Mulberry32, distribuicao, sortear
from llmlab.graficos import RESULTADOS, SERIE, TINTA_2, estilo, salvar
from llmlab.modelo import carregar

PROMPTS = ["O banco fica perto do rio e", "A inteligência artificial é", "Ontem à noite, na cidade,"]
TEMPERATURAS = [0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5]
SEMENTES = list(range(20))
NOVOS = 40


def gerar(prompt: str, temperatura: float, sementes: list[int], novos: int = NOVOS):
    tok, modelo = carregar()
    ids = tok(prompt, return_tensors="pt").input_ids.repeat(len(sementes), 1)
    rngs = [Mulberry32(s) for s in sementes]
    saida = [[] for _ in sementes]
    surpresa = [[] for _ in sementes]
    past = None
    entrada = ids
    with torch.no_grad():
        for _ in range(novos):
            out = modelo(input_ids=entrada, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1].double().numpy()
            logp = torch.log_softmax(out.logits[:, -1].double(), dim=-1).numpy()
            prox = []
            for i, rng in enumerate(rngs):
                d = distribuicao(logits[i], temperatura)
                t = int(d.mantidos[0]) if temperatura <= 0 else sortear(d, rng)
                prox.append(t)
                saida[i].append(t)
                surpresa[i].append(-logp[i, t])
            entrada = torch.tensor(prox)[:, None]
    return saida, surpresa


def ngramas(seq: list[int], n: int) -> list[tuple]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def metricas(saida: list[list[int]], surpresa: list[list[float]]) -> dict:
    big = [g for s in saida for g in ngramas(s, 2)]
    rep = []
    for s in saida:
        tri = ngramas(s, 3)
        rep.append(1 - len(set(tri)) / len(tri))
    return {
        "saidas_distintas": len({tuple(s) for s in saida}),
        "distinct2": len(set(big)) / len(big),
        "repeticao_interna": float(np.mean(rep)),
        "surpresa_media": float(np.mean(surpresa)),
    }


def rodar() -> dict:
    tok, _ = carregar()
    linhas, exemplos = [], {}
    for p in PROMPTS:
        for t in TEMPERATURAS:
            saida, surpresa = gerar(p, t, SEMENTES)
            linhas.append({"prompt": p, "temperatura": t, **metricas(saida, surpresa)})
            if p == PROMPTS[0]:
                exemplos[str(t)] = [tok.decode(s) for s in saida[:3]]
    res = {"prompts": PROMPTS, "temperaturas": TEMPERATURAS, "sementes": len(SEMENTES), "novos_tokens": NOVOS,
           "linhas": linhas, "exemplos": exemplos}
    graficos(res)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "temperatura.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def graficos(res: dict) -> None:
    import matplotlib.pyplot as plt

    estilo()
    ts = res["temperaturas"]
    medidas = [
        ("distinct2", f"Diversidade entre {res['sementes']} saídas (distinct-2)", None),
        ("repeticao_interna", "Trigramas repetidos na mesma saída", "%"),
        ("surpresa_media", "Surpresa do modelo (−log p, nats)", None),
    ]
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, (chave, titulo, unidade) in zip(eixos, medidas):
        for k, p in enumerate(res["prompts"]):
            ys = [r[chave] for r in res["linhas"] if r["prompt"] == p]
            if unidade == "%":
                ys = [100 * y for y in ys]
            ax.plot(ts, ys, color=SERIE[k], marker="o", ms=4, label=f"“{p}…”")
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("temperatura")
        ax.set_xticks(ts, [f"{t:g}" for t in ts])
    eixos[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), labelcolor=TINTA_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "temperatura.png")


if __name__ == "__main__":
    r = rodar()
    for linha in r["linhas"]:
        print(f"{linha['prompt'][:22]:22s} T={linha['temperatura']:.1f} distintas {linha['saidas_distintas']:2d} "
              f"d2 {linha['distinct2']:.2f} rep {linha['repeticao_interna']:.2f} surpresa {linha['surpresa_media']:.2f}")
    for t, ex in r["exemplos"].items():
        print(t, "|", ex[0].replace("\n", " ")[:120])
