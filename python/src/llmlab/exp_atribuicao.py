"""Experimento 4: o mapa de atenção explica a previsão?

Protocolo
---------
* 16 frases incompletas; o alvo é o token que o próprio modelo acha mais provável em seguida.
* Para cada token anterior ao último, quatro notas de "importância":
  - atenção: peso que a última posição dá ao token, média das 12 cabeças, em cada camada;
  - rollout de atenção (Abnar & Zuidema, 2020): multiplica as matrizes (média das cabeças + identidade
    para a conexão residual) da camada 1 à 12;
  - gradiente × entrada: norma de (∂ logit do alvo / ∂ embedding) ⊙ embedding;
  - oclusão: quanto cai o log p do alvo quando o token é apagado da frase (a referência: mexe na entrada
    e olha a saída, sem supor nada sobre o mecanismo).
* Concordância: correlação de Spearman entre cada nota e a oclusão, por frase, e média nas 16 frases.
  O último token fica de fora (é a própria posição que prevê).
* Também meço a fração de atenção que vai para o primeiro token ("sumidouro de atenção").
"""

from __future__ import annotations

import json

import numpy as np
import torch
from llmlab.graficos import AZUL_SEQ, RESULTADOS, SERIE, TINTA_2, estilo, salvar
from llmlab.modelo import carregar, legivel
from scipy.stats import spearmanr

FRASES = [
    "O banco fica perto do rio? Não, o banco fica perto da",
    "Paguei a conta no banco e o gerente me deu um",
    "Maria deu o livro para João, e ele agradeceu a",
    "A capital da França é",
    "Sentei no banco da praça e fiquei olhando as",
    "O barco encalhou no banco de",
    "Ele abriu o guarda-chuva porque estava",
    "Os alunos entregaram a prova ao",
    "Depois de pescar no rio, voltei para",
    "O Brasil é o maior país da América do",
    "Ela colocou o leite na geladeira porque estava",
    "O médico receitou um remédio para a",
    "Comprei pão na padaria e café no",
    "O gato subiu no telhado e não conseguia",
    "No inverno faz frio e no verão faz",
    "Depositei o dinheiro no banco para pagar a",
]


def logp_alvo(modelo, ids: list[int], alvo: int) -> float:
    with torch.no_grad():
        lg = modelo(torch.tensor([ids])).logits[0, -1]
    return float(torch.log_softmax(lg, -1)[alvo])


def notas(frase: str) -> dict:
    tok, modelo = carregar()
    ids = tok(frase).input_ids
    T = len(ids)
    with torch.no_grad():
        out = modelo(torch.tensor([ids]), output_attentions=True)
    alvo = int(out.logits[0, -1].argmax())
    att = torch.stack(out.attentions)[:, 0].numpy()  # [L, H, T, T]
    media = att.mean(axis=1)  # [L, T, T]
    atencao = media[:, -1, :]  # [L, T]

    roll = np.eye(T)
    for camada in media:
        a = 0.5 * camada + 0.5 * np.eye(T)
        a = a / a.sum(axis=1, keepdims=True)
        roll = a @ roll
    rollout = roll[-1]

    with torch.enable_grad():
        emb = modelo.transformer.wte(torch.tensor([ids])).detach().requires_grad_(True)
        logit = modelo(inputs_embeds=emb).logits[0, -1, alvo]
        logit.backward()
    gxi = (emb.grad[0] * emb[0]).norm(dim=-1).detach().numpy()

    base = logp_alvo(modelo, ids, alvo)
    oclusao = np.array([base - logp_alvo(modelo, ids[:i] + ids[i + 1 :], alvo) for i in range(T - 1)] + [np.nan])
    return {
        "frase": frase,
        "tokens": [legivel(tok.decode([i])) for i in ids],
        "alvo": tok.decode([alvo]),
        "atencao": atencao,
        "rollout": rollout,
        "gxi": gxi,
        "oclusao": oclusao,
        "sumidouro": atencao[:, 0],
    }


def rho(a: np.ndarray, b: np.ndarray) -> float:
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def rodar() -> dict:
    todas = [notas(f) for f in FRASES]
    L = todas[0]["atencao"].shape[0]
    por_camada, por_camada_sem0 = [], []
    for c in range(L):
        por_camada.append([rho(n["atencao"][c, :-1], n["oclusao"][:-1]) for n in todas])
        por_camada_sem0.append([rho(n["atencao"][c, 1:-1], n["oclusao"][1:-1]) for n in todas])
    rollout = [rho(n["rollout"][:-1], n["oclusao"][:-1]) for n in todas]
    gxi = [rho(n["gxi"][:-1], n["oclusao"][:-1]) for n in todas]
    # top-1: o token mais importante pela oclusão é o mesmo que recebe mais atenção?
    top1_att = [int(np.argmax(n["atencao"][-1, :-1]) == np.nanargmax(n["oclusao"])) for n in todas]
    top1_gxi = [int(np.argmax(n["gxi"][:-1]) == np.nanargmax(n["oclusao"])) for n in todas]
    sumid = np.array([n["sumidouro"] for n in todas])  # [frases, L]
    imp0 = [float(n["oclusao"][0] / np.nansum(np.abs(n["oclusao"]))) for n in todas]
    res = {
        "n_frases": len(FRASES),
        "spearman_atencao_por_camada": [float(np.mean(x)) for x in por_camada],
        "spearman_atencao_por_camada_sem_1o_token": [float(np.mean(x)) for x in por_camada_sem0],
        "spearman_rollout": float(np.mean(rollout)),
        "spearman_gradiente_x_entrada": float(np.mean(gxi)),
        "top1_igual_oclusao": {"atencao_ultima_camada": float(np.mean(top1_att)), "gradiente_x_entrada": float(np.mean(top1_gxi))},
        "atencao_no_1o_token_por_camada": sumid.mean(axis=0).tolist(),
        "oclusao_peso_relativo_1o_token": float(np.mean(imp0)),
        "exemplos": [
            {"frase": n["frase"], "alvo": n["alvo"], "tokens": n["tokens"],
             "atencao_ultima_camada": n["atencao"][-1].tolist(), "rollout": n["rollout"].tolist(),
             "gradiente_x_entrada": n["gxi"].tolist(), "oclusao": [None if np.isnan(v) else float(v) for v in n["oclusao"]]}
            for n in todas
        ],
    }
    graficos(res, por_camada, rollout, gxi, todas[0])
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "atribuicao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def graficos(res: dict, por_camada, rollout, gxi, ex: dict) -> None:
    import matplotlib.pyplot as plt

    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    cam = np.arange(1, len(por_camada) + 1)
    m = np.array([np.mean(x) for x in por_camada])
    se = np.array([np.std(x, ddof=1) / np.sqrt(len(x)) for x in por_camada])
    a1.fill_between(cam, m - 1.96 * se, m + 1.96 * se, color=SERIE[0], alpha=0.15, lw=0)
    a1.plot(cam, m, color=SERIE[0], marker="o", ms=4)
    a1.text(cam[-1] + 0.3, m[-1], "atenção da camada", color=TINTA_2, va="center", fontsize=9)
    a1.hlines(np.mean(gxi), 1, cam[-1], color=SERIE[1], lw=2)
    a1.text(cam[-1] + 0.3, np.mean(gxi), "gradiente × entrada", color=TINTA_2, va="center", fontsize=9)
    a1.hlines(np.mean(rollout), 1, cam[-1], color=SERIE[2], lw=2, ls="--")
    a1.text(cam[-1] + 0.3, np.mean(rollout), "rollout de atenção", color=TINTA_2, va="center", fontsize=9)
    a1.axhline(0, color="#b9b8b3", lw=1)
    a1.set_xlim(0.6, cam[-1] + 4.2)
    a1.set_ylim(-0.6, 1)
    a1.set_xticks(cam)
    a1.set_xlabel("camada")
    a1.set_ylabel("Spearman com a oclusão")
    a1.set_title("Concordância com a oclusão (média de 16 frases)")

    linhas = [("atenção (camada 12)", ex["atencao"][-1]), ("rollout", ex["rollout"]),
              ("gradiente × entrada", ex["gxi"]), ("oclusão", np.clip(ex["oclusao"], 0, None))]
    M = []
    for _, v in linhas:
        v = np.array(v[:-1], dtype=float)
        M.append(v / (np.nanmax(np.abs(v)) or 1))
    M = np.array(M)
    a2.imshow(M, cmap=AZUL_SEQ, vmin=0, vmax=1, aspect="auto")
    a2.set_yticks(range(len(linhas)), [n for n, _ in linhas])
    a2.set_xticks(range(M.shape[1]), ex["tokens"][:-1], rotation=55, ha="right", fontsize=9)
    a2.grid(False)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title(f"“{ex['frase']}” → “{ex['alvo'].strip()}”", fontsize=10.5)
    fig.tight_layout()
    salvar(fig, "atencao_x_atribuicao.png")


if __name__ == "__main__":
    r = rodar()
    print("spearman atenção por camada:", [round(x, 2) for x in r["spearman_atencao_por_camada"]])
    print("sem 1o token:", [round(x, 2) for x in r["spearman_atencao_por_camada_sem_1o_token"]])
    print("rollout", round(r["spearman_rollout"], 2), "gxi", round(r["spearman_gradiente_x_entrada"], 2))
    print("top1", r["top1_igual_oclusao"])
    print("atenção no 1o token:", [round(x, 2) for x in r["atencao_no_1o_token_por_camada"]])
    print("oclusão rel 1o token", round(r["oclusao_peso_relativo_1o_token"], 3))
    for e in r["exemplos"][:4]:
        print(e["frase"], "->", e["alvo"])
