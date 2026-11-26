// Contas com vetores, sem dependências (testadas em web/test).

// Linha `id` da matriz int8 [V, D] com escala por linha -> Float32Array [D].
export function embeddingDe(q, escala, D, id) {
  const v = new Float32Array(D);
  const s = escala[id];
  const o = id * D;
  for (let d = 0; d < D; d++) v[d] = q[o + d] * s;
  return v;
}

// logits[v] = <h, linha v> = escala[v] * Σ q[v,d] h[d]   (pesos amarrados do GPT-2)
export function logitsDeEstado(q, escala, D, h) {
  const V = escala.length;
  const out = new Float32Array(V);
  for (let v = 0; v < V; v++) {
    let s = 0;
    const o = v * D;
    for (let d = 0; d < D; d++) s += q[o + d] * h[d];
    out[v] = s * escala[v];
  }
  return out;
}

// PCA 2D por iteração de potência (determinística). X: array de Float32Array, todos com a mesma dimensão.
// Os eixos são ajustados só nos `nAjuste` primeiros pontos (padrão: todos); os demais são projetados neles.
// Devolve as coordenadas [n][2] e a fração da variância (dos pontos de ajuste) explicada por cada eixo.
export function pca2(X, iter = 200, nAjuste = X.length) {
  const n = nAjuste;
  const D = X[0].length;
  const media = new Float64Array(D);
  for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) media[d] += X[i][d] / n;
  const todos = X.map((x) => { const c = new Float64Array(D); for (let d = 0; d < D; d++) c[d] = x[d] - media[d]; return c; });
  const C = todos.slice(0, n);
  let total = 0;
  for (const c of C) for (let d = 0; d < D; d++) total += c[d] * c[d];
  const eixos = [];
  const variancias = [];
  for (let k = 0; k < 2; k++) {
    let v = new Float64Array(D);
    for (let d = 0; d < D; d++) v[d] = Math.sin(d * 12.9898 + k * 78.233); // vetor inicial fixo
    for (let it = 0; it < iter; it++) {
      const w = new Float64Array(D);
      for (const c of C) {
        let p = 0;
        for (let d = 0; d < D; d++) p += c[d] * v[d];
        for (let d = 0; d < D; d++) w[d] += p * c[d];
      }
      for (const e of eixos) { let p = 0; for (let d = 0; d < D; d++) p += w[d] * e[d]; for (let d = 0; d < D; d++) w[d] -= p * e[d]; }
      let nrm = 0;
      for (let d = 0; d < D; d++) nrm += w[d] * w[d];
      nrm = Math.sqrt(nrm) || 1;
      for (let d = 0; d < D; d++) w[d] /= nrm;
      v = w;
    }
    // orienta o sinal de forma estável: maior componente positiva
    let iMax = 0;
    for (let d = 1; d < D; d++) if (Math.abs(v[d]) > Math.abs(v[iMax])) iMax = d;
    if (v[iMax] < 0) for (let d = 0; d < D; d++) v[d] = -v[d];
    eixos.push(v);
    let var_ = 0;
    for (const c of C) { let p = 0; for (let d = 0; d < D; d++) p += c[d] * v[d]; var_ += p * p; }
    variancias.push(total ? var_ / total : 0);
  }
  const coords = todos.map((c) => eixos.map((e) => { let p = 0; for (let d = 0; d < D; d++) p += c[d] * e[d]; return p; }));
  return { coords, variancia: variancias };
}

export function cosseno(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return ab / (Math.sqrt(aa * bb) || 1);
}

// Acerto do "centroide mais próximo" deixando um de fora (mesma conta de exp_banco.py):
// centraliza, normaliza e compara cada vetor com o centroide de cada rótulo calculado sem ele.
export function acuraciaCentroide(X, rotulos) {
  const n = X.length;
  const D = X[0].length;
  const media = new Float64Array(D);
  for (const x of X) for (let d = 0; d < D; d++) media[d] += x[d] / n;
  const C = X.map((x) => {
    const c = new Float64Array(D);
    let s = 0;
    for (let d = 0; d < D; d++) { c[d] = x[d] - media[d]; s += c[d] * c[d]; }
    s = Math.sqrt(s) || 1;
    for (let d = 0; d < D; d++) c[d] /= s;
    return c;
  });
  const classes = [...new Set(rotulos)];
  let acertos = 0;
  for (let i = 0; i < n; i++) {
    let melhor = null, melhorSim = -Infinity;
    for (const r of classes) {
      const cent = new Float64Array(D);
      for (let j = 0; j < n; j++) if (j !== i && rotulos[j] === r) for (let d = 0; d < D; d++) cent[d] += C[j][d];
      const sim = cosseno(C[i], cent);
      if (sim > melhorSim) { melhorSim = sim; melhor = r; }
    }
    if (melhor === rotulos[i]) acertos++;
  }
  return acertos / n;
}
