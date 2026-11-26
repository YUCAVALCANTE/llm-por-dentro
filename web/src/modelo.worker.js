// Web Worker do modelo: baixa (com cache) e roda o GPT-2 português fora da thread principal.
// O ONNX recebe os embeddings de entrada e devolve os estados de todas as camadas e toda a atenção.
// A matriz de embeddings (wte) fica aqui em int8 e serve duas vezes: na entrada (ID -> vetor) e na
// saída (logits = último estado · cada linha da matriz), porque o GPT-2 amarra os dois pesos.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';
import { logitsDeEstado, embeddingDe } from './vetores.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = 'llm-por-dentro-modelo-v1';
let sessao = null;
let cfg = null;
let wte = null; // Int8Array [V * D]
let escala = null; // Float32Array [V]

async function baixar(url, tamanho, aoProgresso) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* sem Cache API (ex.: aba privada) */ }
  if (cache) {
    const r = await cache.match(url);
    if (r) { const b = await r.arrayBuffer(); aoProgresso(b.byteLength, true); return b; }
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`falha ao baixar ${url} (${resp.status})`);
  const total = Number(resp.headers.get('content-length')) || tamanho;
  const leitor = resp.body.getReader();
  const pedacos = [];
  let recebidos = 0;
  for (;;) {
    const { done, value } = await leitor.read();
    if (done) break;
    pedacos.push(value);
    recebidos += value.byteLength;
    aoProgresso(value.byteLength, false, total);
  }
  const buf = new Uint8Array(recebidos);
  let o = 0;
  for (const p of pedacos) { buf.set(p, o); o += p.byteLength; }
  if (cache) { try { await cache.put(url, new Response(buf.slice().buffer)); } catch { /* cota cheia */ } }
  return buf.buffer;
}

async function carregar({ base }) {
  const t0 = performance.now();
  cfg = await (await fetch(`${base}modelo.json`)).json();
  const nomes = ['blocos.onnx', 'wte.i8', 'wte.escala.f32'];
  const total = nomes.reduce((s, n) => s + cfg.arquivos[n].bytes, 0);
  let recebidos = 0;
  let doCache = true;
  const progresso = (n, cache) => {
    recebidos += n;
    if (!cache) doCache = false;
    self.postMessage({ tipo: 'progresso', recebidos, total });
  };
  const [onnx, w, e] = await Promise.all(nomes.map((n) => baixar(`${base}${n}`, cfg.arquivos[n].bytes, progresso)));
  wte = new Int8Array(w);
  escala = new Float32Array(e);
  self.postMessage({ tipo: 'etapa', texto: 'Preparando o modelo…' });
  // disable_quant_qdq: deixa o onnxruntime "dobrar" os DequantizeLinear dos pesos uma vez, na carga,
  // em vez de desquantizar 85 milhões de pesos a cada passada (fica ~2x mais rápido).
  sessao = await ort.InferenceSession.create(onnx, {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all', extra: { session: { disable_quant_qdq: '1' } },
  });
  return { cfg, doCache, ms: performance.now() - t0, threads: ort.env.wasm.numThreads };
}

async function passar({ ids, semLogits = false }) {
  const D = cfg.dimensao;
  const T = ids.length;
  const emb = new Float32Array(T * D);
  for (let t = 0; t < T; t++) emb.set(embeddingDe(wte, escala, D, ids[t]), t * D);
  const t0 = performance.now();
  const saida = await sessao.run({ inputs_embeds: new ort.Tensor('float32', emb, [1, T, D]) });
  const t1 = performance.now();
  const estados = saida.estados.data;
  const atencao = saida.atencao.data;
  const ultimo = estados.subarray((cfg.camadas * T + (T - 1)) * D, (cfg.camadas * T + T) * D);
  const logits = semLogits ? new Float32Array(0) : logitsDeEstado(wte, escala, D, ultimo);
  const t2 = performance.now();
  return { estados, atencao, logits, T, ms: { rede: t1 - t0, logits: t2 - t1 } };
}

function embeddings({ grupos }) {
  // média dos embeddings estáticos de cada grupo de IDs (uma palavra pode ter vários tokens)
  const D = cfg.dimensao;
  return grupos.map((ids) => {
    const v = new Float32Array(D);
    for (const id of ids) { const e = embeddingDe(wte, escala, D, id); for (let d = 0; d < D; d++) v[d] += e[d] / ids.length; }
    return v;
  });
}

self.onmessage = async ({ data }) => {
  const { id, tipo } = data;
  try {
    if (tipo === 'carregar') self.postMessage({ id, ok: true, r: await carregar(data) });
    else if (tipo === 'passar') {
      const r = await passar(data);
      self.postMessage({ id, ok: true, r }, [r.estados.buffer, r.atencao.buffer, r.logits.buffer]);
    } else if (tipo === 'embeddings') self.postMessage({ id, ok: true, r: embeddings(data) });
  } catch (e) {
    self.postMessage({ id, ok: false, erro: String(e?.message || e) });
  }
};
