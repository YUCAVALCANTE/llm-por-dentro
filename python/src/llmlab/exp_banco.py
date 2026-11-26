"""Experimento 1: em que camada os sentidos de "banco" se separam?

Protocolo
---------
* 36 frases curtas, 12 por sentido (instituição financeira, assento, banco de areia/margem de rio),
  todas terminando em "banco": o contexto que decide o sentido vem ANTES da palavra, porque o GPT-2
  é causal (cada posição só enxerga as anteriores).
* Para cada camada (0 = embedding + posição, 1..12 = saída de cada bloco) pego o vetor da posição de
  " banco" e centralizo (subtraio a média dos 36 vetores daquela camada). Sem centralizar, todos os
  vetores do GPT-2 apontam quase para a mesma direção e o cosseno fica perto de 1 para tudo.
* Medidas por camada: cosseno médio dentro do mesmo sentido x entre sentidos; acurácia de
  "centroide mais próximo" deixando uma frase de fora (leave-one-out).
* Controle causal: "O banco fica perto do rio?" e "O banco aprovou o empréstimo." têm o mesmo vetor
  em " banco" em todas as camadas, porque o que vem depois não entra.
"""

from __future__ import annotations

import json

import numpy as np
from llmlab.graficos import FIGURAS, RESULTADOS, SERIE, TINTA_2, estilo, salvar
from llmlab.modelo import carregar, passar

FRASES = {
    "financeiro": [
        "Fui sacar dinheiro no banco",
        "Pedi um empréstimo ao banco",
        "Depositei o salário no banco",
        "Paguei os juros cobrados pelo banco",
        "Abri uma conta corrente no banco",
        "O gerente da agência do banco",
        "Paguei o boleto no caixa do banco",
        "O cartão de crédito emitido pelo banco",
        "Transferi as economias para outro banco",
        "Renegociei a dívida com o banco",
        "Investi nas ações do maior banco",
        "Os ladrões assaltaram o cofre do banco",
    ],
    "assento": [
        "Cansado da caminhada, sentei num banco",
        "Na praça arborizada havia um velho banco",
        "O carpinteiro lixou e pintou o banco",
        "As crianças se sentaram lado a lado no banco",
        "No jardim, perto das flores, colocamos um banco",
        "O mendigo dormiu deitado no banco",
        "Comprei uma mesa de jantar e um banco",
        "O pássaro pousou no encosto do banco",
        "Ela se levantou devagar do banco",
        "No parque, o casal namorava sentado no banco",
        "O marceneiro serrou as pernas do banco",
        "Esperei o ônibus sentado no banco",
    ],
    "rio": [
        "O barco encalhou num banco",
        "Na maré baixa, surgiu no meio do rio um banco",
        "As garças pousaram na areia do banco",
        "O pescador ancorou a canoa junto ao banco",
        "A correnteza do rio formou um novo banco",
        "Na foz do rio, a lancha bateu num banco",
        "As tartarugas desovam na areia de um banco",
        "O piloto da balsa desviou do banco",
        "Com a seca, o rio virou um enorme banco",
        "O navio ficou preso num traiçoeiro banco",
        "No delta, cada ilha nasceu de um banco",
        "O canoeiro arrastou o barco pelo banco",
    ],
}

AMBIGUAS = [
    "Na margem do rio, ao lado da ponte, fica o banco",
    "Sentado à beira do rio, olhei para o banco",
    "Depois de pescar no rio, fui ao banco",
]


def posicao_banco(tokens: list[str]) -> int:
    alvo = [i for i, t in enumerate(tokens) if t.lstrip("Ġ").lower() == "banco"]
    if not alvo:
        raise ValueError(f"'banco' não é um token inteiro em {tokens}")
    return alvo[-1]


def vetores(frases: list[str]) -> np.ndarray:
    """[N, camadas+1, D]: o estado na posição de ' banco' em cada camada."""
    tok, modelo = carregar()
    vs = []
    for f in frases:
        p = passar(f, tok, modelo)
        vs.append(p.estados[:, posicao_banco(p.tokens)])
    return np.stack(vs)


def cosseno_centralizado(v: np.ndarray) -> np.ndarray:
    c = v - v.mean(axis=0, keepdims=True)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return c @ c.T


def acuracia_loo(v: np.ndarray, rotulos: np.ndarray) -> float:
    c = v - v.mean(axis=0, keepdims=True)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    acertos = 0
    for i in range(len(c)):
        resto = np.arange(len(c)) != i
        cents = {r: c[resto & (rotulos == r)].mean(axis=0) for r in np.unique(rotulos)}
        pred = max(cents, key=lambda r: c[i] @ cents[r] / np.linalg.norm(cents[r]))
        acertos += pred == rotulos[i]
    return acertos / len(c)


def rodar() -> dict:
    sentidos = list(FRASES)
    frases = [f for s in sentidos for f in FRASES[s]]
    rotulos = np.array([s for s in sentidos for _ in FRASES[s]])
    V = vetores(frases)  # [N, L+1, D]
    n_camadas = V.shape[1]

    por_camada = []
    mesmo = rotulos[:, None] == rotulos[None, :]
    fora_diag = ~np.eye(len(rotulos), dtype=bool)
    for c in range(n_camadas):
        S = cosseno_centralizado(V[:, c])
        dentro = float(S[mesmo & fora_diag].mean())
        entre = float(S[~mesmo].mean())
        pares = {}
        for i, a in enumerate(sentidos):
            for b in sentidos[i + 1 :]:
                pares[f"{a}×{b}"] = float(S[np.ix_(rotulos == a, rotulos == b)].mean())
        por_camada.append(
            {"camada": c, "dentro": dentro, "entre": entre, "separacao": dentro - entre,
             "acuracia_loo": acuracia_loo(V[:, c], rotulos), "pares": pares}
        )

    # Frases ambíguas: cosseno (centralizado com a média das 36) com o centroide de cada sentido.
    A = vetores(AMBIGUAS)
    ambiguas = []
    for f, a in zip(AMBIGUAS, A):
        linha = []
        for c in range(n_camadas):
            m = V[:, c].mean(axis=0)
            x = a[c] - m
            sims = {}
            for s in sentidos:
                cent = V[rotulos == s, c].mean(axis=0) - m
                sims[s] = float(x @ cent / (np.linalg.norm(x) * np.linalg.norm(cent)))
            linha.append(sims)
        ambiguas.append({"frase": f, "por_camada": linha})

    # Controle causal
    tok, modelo = carregar()
    p1 = passar("O banco fica perto do rio?", tok, modelo)
    p2 = passar("O banco aprovou o empréstimo.", tok, modelo)
    i1, i2 = posicao_banco(p1.tokens), posicao_banco(p2.tokens)
    dif_max = float(np.abs(p1.estados[:, i1] - p2.estados[:, i2]).max())

    melhor = max(por_camada, key=lambda r: r["separacao"])
    primeira_100 = next((r["camada"] for r in por_camada if r["acuracia_loo"] >= 0.999), None)
    res = {
        "n_frases": len(frases),
        "sentidos": sentidos,
        "por_camada": por_camada,
        "camada_maior_separacao": melhor["camada"],
        "primeira_camada_acuracia_100": primeira_100,
        "ambiguas": ambiguas,
        "controle_causal": {
            "frases": ["O banco fica perto do rio?", "O banco aprovou o empréstimo."],
            "diferenca_maxima_no_vetor_de_banco": dif_max,
        },
    }
    graficos(res, V, rotulos)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "banco.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def graficos(res: dict, V: np.ndarray, rotulos: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    estilo()
    cam = [r["camada"] for r in res["por_camada"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    a1.plot(cam, [r["dentro"] for r in res["por_camada"]], color=SERIE[0], marker="o", ms=4)
    a1.plot(cam, [r["entre"] for r in res["por_camada"]], color=SERIE[1], marker="o", ms=4)
    a1.text(cam[-1] + 0.25, res["por_camada"][-1]["dentro"], "mesmo sentido", color=TINTA_2, va="center")
    a1.text(cam[-1] + 0.25, res["por_camada"][-1]["entre"], "sentidos diferentes", color=TINTA_2, va="center")
    a1.axhline(0, color="#b9b8b3", lw=1)
    a1.set_xlim(-0.3, cam[-1] + 3.6)
    a1.set_xticks(cam)
    a1.set_xlabel("camada (0 = embedding + posição)")
    a1.set_ylabel("cosseno médio (centralizado)")
    a1.set_title('Vetor de "banco": mesmo sentido × sentidos diferentes')

    a2.plot(cam, [100 * r["acuracia_loo"] for r in res["por_camada"]], color=SERIE[0], marker="o", ms=4)
    a2.axhline(100 / 3, color="#b9b8b3", lw=1, ls="--")
    a2.text(0, 100 / 3 + 2, "acaso (3 sentidos)", color=TINTA_2, fontsize=9)
    a2.set_ylim(0, 105)
    a2.set_xticks(cam)
    a2.set_xlabel("camada")
    a2.set_ylabel("% de acerto")
    a2.set_title("Sentido certo pelo centroide mais próximo")
    fig.tight_layout()
    salvar(fig, "banco_camadas.png")

    # Projeção 2D (PCA) em três camadas: só ilustração do que os números acima medem.
    camadas = [0, 4, res["camada_maior_separacao"]]
    camadas = sorted(set(camadas))
    fig, eixos = plt.subplots(1, len(camadas), figsize=(4 * len(camadas), 3.8))
    for ax, c in zip(eixos, camadas):
        X = V[:, c] - V[:, c].mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        P = X @ vt[:2].T
        for k, s in enumerate(res["sentidos"]):
            m = rotulos == s
            ax.scatter(P[m, 0], P[m, 1], s=42, color=SERIE[k], edgecolor="white", linewidth=1.2, label=s, zorder=3)
        ax.set_title(f"camada {c}")
        ax.set_xticks([])
        ax.set_yticks([])
    eixos[0].legend(loc="best", fontsize=9)
    fig.suptitle('Projeção 2D (PCA) dos 36 vetores de "banco". É uma projeção: distâncias são aproximadas.',
                 x=0.01, ha="left", fontsize=10, color=TINTA_2)
    fig.tight_layout()
    salvar(fig, "banco_pca.png")


if __name__ == "__main__":
    r = rodar()
    for linha in r["por_camada"]:
        print(f"camada {linha['camada']:2d}  dentro {linha['dentro']:+.3f}  entre {linha['entre']:+.3f}  "
              f"acc {linha['acuracia_loo']:.2f}")
    print("controle causal:", r["controle_causal"])
    print("figuras em", FIGURAS)
