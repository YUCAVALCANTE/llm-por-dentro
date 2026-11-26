import numpy as np
import pytest
from llmlab.amostragem import Mulberry32, distribuicao, proximo_token, softmax

LOGITS = [2.0, 1.0, 0.5, 0.5, -1.0]


def test_mulberry32_igual_ao_js():
    r = Mulberry32(42)
    # mesmos valores em web/test/amostragem.test.js
    assert [r(), r(), r()] == [0.6011037519201636, 0.44829055899754167, 0.8524657934904099]


def test_softmax_estavel_e_normalizado():
    p = softmax([1000.0, 999.0, 0.0])
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_temperatura_zero_e_alta():
    assert softmax(LOGITS, 0).tolist() == [1, 0, 0, 0, 0]
    quente = softmax(LOGITS, 100)
    assert quente.max() - quente.min() < 0.01


def test_top_k_top_p_iguais_ao_js():
    d = distribuicao(LOGITS, 0.7, 3, 0.9)
    assert d.mantidos.tolist() == [0, 1, 2]
    assert d.finais.sum() == pytest.approx(1.0)
    assert d.finais[0] == pytest.approx(0.736935857641613, abs=1e-12)
    assert distribuicao(LOGITS, top_k=1).mantidos.tolist() == [0]
    assert len(distribuicao(LOGITS, top_p=0.01).mantidos) == 1


def test_empate_menor_id_primeiro():
    assert distribuicao([0, 3, 3, 1], top_k=2).mantidos.tolist() == [1, 2]


def test_sequencia_com_semente_igual_ao_js():
    g = Mulberry32(7)
    assert [proximo_token(LOGITS, 1.2, rng=g) for _ in range(10)] == [0, 0, 4, 2, 1, 0, 0, 0, 1, 2]


def test_frequencias_batem_com_probabilidades():
    g = Mulberry32(1)
    n = 20000
    cont = np.bincount([proximo_token(LOGITS, 1.0, rng=g) for _ in range(n)], minlength=5)
    assert np.abs(cont / n - softmax(LOGITS)).max() < 0.015
