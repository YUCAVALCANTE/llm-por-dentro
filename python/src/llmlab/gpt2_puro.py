"""O GPT-2 inteiro em poucas linhas de PyTorch, usando os pesos do modelo do Hugging Face.

Serve a dois propósitos:
1. exportar para ONNX um grafo limpo, que devolve estados de todas as camadas e todos os pesos de atenção;
2. mostrar que não há mágica: embedding + posição, e 12 vezes [LayerNorm -> atenção causal -> soma;
   LayerNorm -> MLP -> soma]. Os testes conferem que o resultado bate com o `transformers`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class GPT2Puro(nn.Module):
    def __init__(self, hf_modelo):
        super().__init__()
        t = hf_modelo.transformer
        cfg = hf_modelo.config
        self.n_cabecas = cfg.n_head
        self.eps = cfg.layer_norm_epsilon
        self.wpe = nn.Parameter(t.wpe.weight.detach().clone())
        self.blocos = nn.ModuleList()
        for b in t.h:
            m = nn.Module()
            m.ln1 = b.ln_1
            m.ln2 = b.ln_2
            # Conv1D do GPT-2 guarda o peso como [entrada, saída]: y = x @ W + b
            m.attn_w, m.attn_b = b.attn.c_attn.weight, b.attn.c_attn.bias
            m.proj_w, m.proj_b = b.attn.c_proj.weight, b.attn.c_proj.bias
            m.fc_w, m.fc_b = b.mlp.c_fc.weight, b.mlp.c_fc.bias
            m.out_w, m.out_b = b.mlp.c_proj.weight, b.mlp.c_proj.bias
            self.blocos.append(m)
        self.ln_f = t.ln_f

    def forward(self, inputs_embeds: torch.Tensor):
        """inputs_embeds [1, T, D] -> (estados [L+1, T, D], atencao [L, H, T, T]).

        Igual ao `output_hidden_states` do transformers: estados[0] = embedding + posição,
        estados[i] = saída do bloco i, e o último já passou pela LayerNorm final.
        """
        x = inputs_embeds[0]
        T, D = x.shape
        H = self.n_cabecas
        dh = D // H
        pos = torch.arange(T)
        h = x + self.wpe[pos]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        estados, atencoes = [h], []
        for i, b in enumerate(self.blocos):
            a = b.ln1(h)
            q, k, v = (a @ b.attn_w + b.attn_b).split(D, dim=-1)
            q = q.reshape(T, H, dh).transpose(0, 1)
            k = k.reshape(T, H, dh).transpose(0, 1)
            v = v.reshape(T, H, dh).transpose(0, 1)
            s = (q @ k.transpose(1, 2)) / math.sqrt(dh)  # [H, T, T]
            s = s.masked_fill(~causal, -1e9)  # máscara causal: ninguém olha para frente
            p = torch.softmax(s, dim=-1)
            atencoes.append(p)
            o = (p @ v).transpose(0, 1).reshape(T, D)
            h = h + (o @ b.proj_w + b.proj_b)
            m = F.gelu(b.ln2(h) @ b.fc_w + b.fc_b, approximate="tanh")
            h = h + (m @ b.out_w + b.out_b)
            estados.append(self.ln_f(h) if i == len(self.blocos) - 1 else h)
        return torch.stack(estados), torch.stack(atencoes)
