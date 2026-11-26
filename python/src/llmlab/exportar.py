"""Exporta o modelo para a demo no navegador.

O ONNX padrão do Transformers.js não devolve os pesos de atenção nem os estados de cada camada.
Por isso a exportação aqui é própria:

* `blocos.onnx` recebe os *embeddings de entrada* `[1, T, D]` e devolve
  `estados [camadas+1, T, D]` (o último já passou pela LayerNorm final) e
  `atencao [camadas, cabeças, T, T]`. Pesos quantizados em int8 (quantização dinâmica do onnxruntime).
* `wte.i8` + `wte.escala.f32`: a matriz de embeddings de tokens `[V, D]` em int8 com uma escala por linha.
  No GPT-2 ela serve duas vezes (pesos amarrados): na entrada, cada ID vira uma linha dela; na saída,
  os logits são o produto do último estado por todas as linhas. A demo faz as duas contas em JavaScript,
  e assim a matriz só é baixada uma vez.
* tokenizadores (`tokenizer.json`) para o Transformers.js: o do modelo e dois para comparação.

Uso: python -m llmlab.exportar --saida web/modelo
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from llmlab.gpt2_puro import GPT2Puro
from llmlab.modelo import MODELO, REVISAO, carregar

TOKENIZADORES = {
    # nome curto: (repositório no Hugging Face, revisão, rótulo)
    "gpt2-pt": (MODELO, REVISAO, "GPT-2 português (o modelo da demo)"),
    "gpt2": ("openai-community/gpt2", None, "GPT-2 original (treinado em inglês)"),
    "qwen2.5": ("Qwen/Qwen2.5-0.5B", None, "Qwen2.5 (multilíngue, 2024)"),
}


def quantizar_linhas(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 simétrico por linha: w ≈ q * escala[:, None]."""
    escala = np.abs(w).max(axis=1) / 127.0
    escala[escala == 0] = 1.0
    q = np.clip(np.round(w / escala[:, None]), -127, 127).astype(np.int8)
    return q, escala.astype(np.float32)


def exportar_onnx(modelo, destino: Path, opset: int = 17) -> None:
    blocos = GPT2Puro(modelo).eval()
    d = modelo.config.n_embd
    exemplo = torch.randn(1, 6, d)
    torch.onnx.export(
        blocos,
        (exemplo,),
        str(destino),
        input_names=["inputs_embeds"],
        output_names=["estados", "atencao"],
        dynamic_axes={"inputs_embeds": {1: "T"}, "estados": {1: "T"}, "atencao": {2: "T", 3: "T"}},
        opset_version=opset,
        dynamo=False,
    )


def quantizar_onnx(origem: Path, destino: Path, minimo: int = 4096) -> None:
    """Quantização só dos pesos: cada matriz de MatMul vira int8 com uma escala por coluna e um
    DequantizeLinear na frente. As ativações continuam em float32, então a atenção quase não muda
    (a quantização dinâmica do onnxruntime, que também quantiza as ativações, distorcia os pesos de
    atenção em até 0,4 neste modelo)."""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(origem))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    novos_nos, remover = [], set()
    for no in g.node:
        if no.op_type != "MatMul":
            continue
        nome = no.input[1]
        if nome not in inits or nome in remover:
            continue
        w = numpy_helper.to_array(inits[nome])
        if w.ndim != 2 or w.size < minimo:
            continue
        escala = np.abs(w).max(axis=0) / 127.0
        escala[escala == 0] = 1.0
        q = np.clip(np.round(w / escala), -127, 127).astype(np.int8)
        g.initializer.extend([
            numpy_helper.from_array(q, nome + "_q"),
            numpy_helper.from_array(escala.astype(np.float32), nome + "_escala"),
            numpy_helper.from_array(np.zeros(w.shape[1], np.int8), nome + "_zero"),
        ])
        novos_nos.append(helper.make_node("DequantizeLinear", [nome + "_q", nome + "_escala", nome + "_zero"],
                                          [nome], axis=1, name=nome + "_dq"))
        remover.add(nome)
    manter = [i for i in g.initializer if i.name not in remover]
    del g.initializer[:]
    g.initializer.extend(manter)
    nos = list(g.node)
    del g.node[:]
    g.node.extend(novos_nos + nos)
    onnx.checker.check_model(m)
    onnx.save(m, str(destino))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def salvar_tokenizadores(saida: Path) -> list[dict]:
    from transformers import AutoTokenizer

    lista = []
    for nome, (repo, rev, rotulo) in TOKENIZADORES.items():
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        pasta = saida / "tokenizadores" / nome
        pasta.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(pasta)
        for extra in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
            (pasta / extra).unlink(missing_ok=True)
        lista.append({"id": nome, "repo": repo, "rotulo": rotulo, "vocabulario": len(tok)})
    return lista


def conferir(saida: Path, tok, modelo, frase: str) -> dict:
    """Roda a versão exportada (int8) e compara com o PyTorch em fp32."""
    import onnxruntime as ort

    ids = tok(frase).input_ids
    q = np.fromfile(saida / "wte.i8", dtype=np.int8).reshape(-1, modelo.config.n_embd)
    esc = np.fromfile(saida / "wte.escala.f32", dtype=np.float32)
    emb = (q[ids].astype(np.float32) * esc[ids, None])[None]
    sess = ort.InferenceSession(str(saida / "blocos.onnx"), providers=["CPUExecutionProvider"])
    estados, atencao = sess.run(None, {"inputs_embeds": emb})
    logits_q = (q.astype(np.float32) @ estados[-1, -1]) * esc
    with torch.no_grad():
        out = modelo(torch.tensor([ids]), output_attentions=True)
    logits = out.logits[0, -1].numpy()
    att = torch.stack(out.attentions)[:, 0].numpy()
    top_ref = np.argsort(-logits)[:10]
    top_q = np.argsort(-logits_q)[:10]
    return {
        "frase": frase,
        "top1_igual": bool(top_ref[0] == top_q[0]),
        "top10_em_comum": int(len(set(top_ref) & set(top_q))),
        "atencao_erro_max": float(np.abs(att - atencao).max()),
        "logits_correlacao": float(np.corrcoef(logits, logits_q)[0, 1]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("web/modelo"))
    args = ap.parse_args(argv)
    saida: Path = args.saida
    saida.mkdir(parents=True, exist_ok=True)

    tok, modelo = carregar()
    cfg = modelo.config
    bruto = saida / "blocos.fp32.onnx"
    exportar_onnx(modelo, bruto)
    quantizar_onnx(bruto, saida / "blocos.onnx")
    bruto.unlink()
    for extra in saida.glob("*.data"):
        extra.unlink()

    q, esc = quantizar_linhas(modelo.transformer.wte.weight.detach().numpy())
    q.tofile(saida / "wte.i8")
    esc.tofile(saida / "wte.escala.f32")

    toks = salvar_tokenizadores(saida)
    verif = [conferir(saida, tok, modelo, f) for f in ("O banco fica perto do rio?", "Depositei o dinheiro no banco e")]
    arquivos = {}
    for nome in ("blocos.onnx", "wte.i8", "wte.escala.f32"):
        p = saida / nome
        arquivos[nome] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifesto = {
        "modelo": MODELO,
        "revisao": REVISAO,
        "licenca": "MIT",
        "camadas": cfg.n_layer,
        "cabecas": cfg.n_head,
        "dimensao": cfg.n_embd,
        "vocabulario": cfg.vocab_size,
        "contexto_max": cfg.n_positions,
        "arquivos": arquivos,
        "tokenizadores": toks,
        "verificacao": verif,
    }
    (saida / "modelo.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
