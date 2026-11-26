"""O GPT-2 "puro" e a exportação ONNX, num GPT-2 minúsculo com pesos aleatórios (sem baixar nada)."""

import numpy as np
import pytest
import torch
from llmlab.exportar import quantizar_linhas
from llmlab.gpt2_puro import GPT2Puro


@pytest.fixture(scope="module")
def mini():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2, attn_implementation="eager")
    return GPT2LMHeadModel(cfg).eval()


def test_gpt2_puro_igual_ao_transformers(mini):
    ids = torch.tensor([[5, 9, 2, 33, 7, 1]])
    with torch.no_grad():
        ref = mini(ids, output_hidden_states=True, output_attentions=True)
        estados, atencao = GPT2Puro(mini)(mini.transformer.wte(ids))
    assert estados.shape == (3, 6, 16)
    assert atencao.shape == (2, 2, 6, 6)
    np.testing.assert_allclose(estados.numpy(), torch.stack(ref.hidden_states)[:, 0].numpy(), atol=1e-5)
    np.testing.assert_allclose(atencao.numpy(), torch.stack(ref.attentions)[:, 0].numpy(), atol=1e-6)
    # pesos amarrados: logits = último estado · wte
    logits = estados[-1] @ mini.transformer.wte.weight.T.detach()
    np.testing.assert_allclose(logits.numpy(), ref.logits[0].numpy(), atol=1e-4)


def test_atencao_causal_e_normalizada(mini):
    with torch.no_grad():
        _, att = GPT2Puro(mini)(torch.randn(1, 7, 16))
    att = att.numpy()
    np.testing.assert_allclose(att.sum(-1), 1.0, atol=1e-6)
    assert np.all(np.triu(att, k=1) < 1e-8)  # ninguém olha para frente


def test_quantizacao_por_linha():
    w = np.random.default_rng(0).normal(size=(50, 16)).astype(np.float32)
    q, esc = quantizar_linhas(w)
    assert q.dtype == np.int8 and esc.shape == (50,)
    assert np.abs(q.astype(np.float32) * esc[:, None] - w).max() <= esc.max() / 2 + 1e-6


def test_exportacao_onnx_formato(mini, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from llmlab.exportar import exportar_onnx, quantizar_onnx

    bruto, final = tmp_path / "b.onnx", tmp_path / "q.onnx"
    exportar_onnx(mini, bruto)
    quantizar_onnx(bruto, final, minimo=16)
    for t in (3, 9):  # comprimento dinâmico
        x = np.random.default_rng(t).normal(size=(1, t, 16)).astype(np.float32)
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_quant_qdq", "1")
        estados, atencao = ort.InferenceSession(str(final), so, providers=["CPUExecutionProvider"]).run(
            None, {"inputs_embeds": x})
        assert estados.shape == (3, t, 16)
        assert atencao.shape == (2, 2, t, t)
        with torch.no_grad():
            e_ref, a_ref = GPT2Puro(mini)(torch.from_numpy(x))
        assert np.abs(atencao - a_ref.numpy()).max() < 0.05
        assert np.abs(estados - e_ref.numpy()).max() < 0.1
