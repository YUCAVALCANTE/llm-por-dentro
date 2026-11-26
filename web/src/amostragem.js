// Logits -> probabilidades -> próximo token. Mesma lógica de python/src/llmlab/amostragem.py.
// 1) logits / temperatura (0 = guloso) · 2) softmax · 3) ordena (empate: menor ID)
// 4) top-k (0 desliga) · 5) top-p (1 desliga) · 6) renormaliza e sorteia com semente (mulberry32).

export function mulberry32(semente) {
  let a = semente >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function softmax(logits, temperatura = 1) {
  const n = logits.length;
  const p = new Float64Array(n);
  let iMax = 0;
  for (let i = 1; i < n; i++) if (logits[i] > logits[iMax]) iMax = i;
  if (temperatura <= 0) { p[iMax] = 1; return p; }
  const m = logits[iMax] / temperatura;
  let soma = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(logits[i] / temperatura - m); p[i] = e; soma += e; }
  for (let i = 0; i < n; i++) p[i] /= soma;
  return p;
}

// Devolve { probs, ordem, mantidos, finais }. `ordem` traz só os `limite` primeiros índices
// (para a interface) além dos mantidos; ordenar os 50 mil é desnecessário.
export function distribuicao(logits, { temperatura = 1, topK = 0, topP = 1, limite = 200 } = {}) {
  const probs = softmax(logits, temperatura);
  const n = probs.length;
  // Pré-filtra candidatos: tudo com prob. acima de um piso pequeno, garantindo ao menos `limite`.
  let cand = [];
  const piso = 1e-7;
  for (let i = 0; i < n; i++) if (probs[i] > piso) cand.push(i);
  if (cand.length < Math.min(limite, n)) cand = Array.from({ length: n }, (_, i) => i);
  cand.sort((a, b) => probs[b] - probs[a] || a - b);
  let k = cand.length;
  if (topK > 0) k = Math.min(k, topK);
  if (topP < 1) {
    let acum = 0, j = 0;
    for (; j < k; j++) { acum += probs[cand[j]]; if (acum >= topP - 1e-12) { j++; break; } }
    k = Math.min(k, Math.max(j, 1));
  }
  k = Math.max(k, 1);
  const mantidos = cand.slice(0, k);
  let soma = 0;
  for (const i of mantidos) soma += probs[i];
  const finais = mantidos.map((i) => probs[i] / soma);
  return { probs, ordem: cand.slice(0, Math.max(limite, k)), mantidos, finais };
}

export function sortear(d, rng) {
  const u = rng();
  let acum = 0;
  for (let j = 0; j < d.mantidos.length; j++) {
    acum += d.finais[j];
    if (u < acum) return d.mantidos[j];
  }
  return d.mantidos[d.mantidos.length - 1];
}

export function proximoToken(logits, opcoes = {}, rng = null) {
  const d = distribuicao(logits, opcoes);
  if ((opcoes.temperatura ?? 1) <= 0 || !rng) return d.mantidos[0];
  return sortear(d, rng);
}
