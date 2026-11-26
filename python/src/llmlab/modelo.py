"""Carregar o modelo e extrair o que a demo mostra: tokens, estados ocultos e atenção."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

MODELO = "pierreguillou/gpt2-small-portuguese"  # GPT-2 small (124M) ajustado em português, licença MIT
REVISAO = "89a916c041b54c8b925e1a3282a5a334684280cb"


@lru_cache(maxsize=2)
def carregar(nome: str = MODELO, revisao: str | None = REVISAO):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nome, revision=revisao)
    # "eager": a implementação que devolve os pesos de atenção (SDPA não devolve).
    modelo = AutoModelForCausalLM.from_pretrained(nome, revision=revisao, attn_implementation="eager")
    modelo.eval()
    torch.set_grad_enabled(False)
    return tok, modelo


@dataclass
class Passagem:
    ids: list[int]
    tokens: list[str]
    estados: np.ndarray  # [camadas + 1, T, D]: 0 = embedding + posição; último = após ln_f
    atencao: np.ndarray  # [camadas, cabeças, T, T]
    logits: np.ndarray  # [T, V]


def passar(texto: str, tok=None, modelo=None) -> Passagem:
    if tok is None or modelo is None:
        tok, modelo = carregar()
    ids = tok(texto, return_tensors="pt").input_ids
    with torch.no_grad():
        out = modelo(ids, output_hidden_states=True, output_attentions=True)
    return Passagem(
        ids=ids[0].tolist(),
        tokens=tok.convert_ids_to_tokens(ids[0]),
        estados=torch.stack(out.hidden_states)[:, 0].numpy(),
        atencao=torch.stack(out.attentions)[:, 0].numpy(),
        logits=out.logits[0].numpy(),
    )


def legivel(token: str) -> str:
    """Texto de um token (já decodificado) com o espaço e a quebra de linha visíveis."""
    return token.replace(" ", "␣").replace("\n", "⏎")
