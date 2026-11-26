from pathlib import Path

import numpy as np
import pytest
from llmlab.exp_banco import acuracia_loo, cosseno_centralizado, posicao_banco
from llmlab.exp_temperatura import metricas, ngramas
from llmlab.exp_tokens import bootstrap_razao, palavras

TOKS = Path(__file__).resolve().parents[2] / "web" / "modelo" / "tokenizadores"


def _tok(nome):
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(TOKS / nome / "tokenizer.json"))


def test_tokenizador_da_demo():
    tok = _tok("gpt2-pt")
    enc = tok.encode("O banco fica perto do rio?")
    assert enc.ids == [47, 6060, 2052, 3066, 298, 2257, 31]
    assert enc.tokens[1] == "Ġbanco"  # o espaço faz parte do token
    assert tok.decode(enc.ids) == "O banco fica perto do rio?"


def test_portugues_custa_mais_no_gpt2_ingles():
    pt, en = "O banco fica perto do rio?", "Is the bank near the river?"
    gpt2, gpt2_pt = _tok("gpt2"), _tok("gpt2-pt")
    assert len(gpt2.encode(pt).ids) > len(gpt2.encode(en).ids)
    assert len(gpt2_pt.encode(pt).ids) < len(gpt2_pt.encode(en).ids)


def test_bytes_ida_e_volta_com_acentos():
    tok = _tok("gpt2")
    texto = "Ação, coração e pão"
    assert tok.decode(tok.encode(texto).ids) == texto


def test_posicao_banco():
    assert posicao_banco(["O", "Ġbanco", "Ġfica"]) == 1
    with pytest.raises(ValueError):
        posicao_banco(["O", "Ġban", "co"])


def test_cosseno_e_acuracia_em_grupos_separados():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 8)) + 5 * np.eye(8)[0]
    b = rng.normal(size=(5, 8)) - 5 * np.eye(8)[0]
    v = np.vstack([a, b])
    rot = np.array(["a"] * 5 + ["b"] * 5)
    S = cosseno_centralizado(v)
    assert S[:5, :5].mean() > S[:5, 5:].mean()
    assert acuracia_loo(v, rot) == 1.0


def test_metricas_de_geracao():
    assert ngramas([1, 2, 3, 4], 2) == [(1, 2), (2, 3), (3, 4)]
    m = metricas([[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]], [[1.0] * 6, [1.0] * 6])
    assert m["saidas_distintas"] == 1
    assert m["repeticao_interna"] == pytest.approx(0.5)
    assert m["surpresa_media"] == 1.0


def test_contagem_e_bootstrap():
    assert palavras("O banco, perto do rio?") == 5
    pt, en = np.array([10, 12, 8, 11]), np.array([8, 9, 7, 9])
    lo, hi = bootstrap_razao(pt, en, n=500)
    assert lo <= pt.sum() / en.sum() <= hi
