// Interface da demo. O modelo roda em src/modelo.worker.js; aqui ficam tokenizadores e desenhos.
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import { distribuicao, sortear, mulberry32 } from './amostragem.js';
import { bytesDoToken, hex, textoDoToken } from './bytes.js';
import { acuraciaCentroide, pca2 } from './vetores.js';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL('../modelo/tokenizadores/', import.meta.url).href;

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const fmt = (x, d = 2) => x.toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => `${fmt(100 * x, d)}%`;
const MAX_TOKENS = 96;

const TOKENIZADORES = [
  { id: 'gpt2-pt', rotulo: 'GPT-2 português (o desta demo)' },
  { id: 'gpt2', rotulo: 'GPT-2 original (inglês)' },
  { id: 'qwen2.5', rotulo: 'Qwen2.5 (multilíngue)' },
];

// As mesmas 36 frases do experimento 1 (12 por sentido), todas terminando em "banco": o contexto vem antes.
const REFERENCIAS = {
  financeiro: [
    "Fui sacar dinheiro no banco",
    "Pedi um empréstimo ao banco",
    "Depositei o salário no banco",
    "Paguei os juros cobrados pelo banco",
    "Abri uma conta corrente no banco",
    "O gerente da agência do banco",
    "Paguei o boleto no caixa do banco",
    "O cartão de crédito emitido pelo banco",
    "Transferi as economias para outro banco",
    "Renegociei a dívida com o banco",
    "Investi nas ações do maior banco",
    "Os ladrões assaltaram o cofre do banco"
  ],
  assento: [
    "Cansado da caminhada, sentei num banco",
    "Na praça arborizada havia um velho banco",
    "O carpinteiro lixou e pintou o banco",
    "As crianças se sentaram lado a lado no banco",
    "No jardim, perto das flores, colocamos um banco",
    "O mendigo dormiu deitado no banco",
    "Comprei uma mesa de jantar e um banco",
    "O pássaro pousou no encosto do banco",
    "Ela se levantou devagar do banco",
    "No parque, o casal namorava sentado no banco",
    "O marceneiro serrou as pernas do banco",
    "Esperei o ônibus sentado no banco"
  ],
  rio: [
    "O barco encalhou num banco",
    "Na maré baixa, surgiu no meio do rio um banco",
    "As garças pousaram na areia do banco",
    "O pescador ancorou a canoa junto ao banco",
    "A correnteza do rio formou um novo banco",
    "Na foz do rio, a lancha bateu num banco",
    "As tartarugas desovam na areia de um banco",
    "O piloto da balsa desviou do banco",
    "Com a seca, o rio virou um enorme banco",
    "O navio ficou preso num traiçoeiro banco",
    "No delta, cada ilha nasceu de um banco",
    "O canoeiro arrastou o barco pelo banco"
  ]
};
const CORES = { financeiro: 'var(--s1)', assento: 'var(--s2)', rio: 'var(--s3)', frase: 'var(--ink)' };
const GRUPOS_ESTATICOS = {
  financeiro: ['dinheiro', 'empréstimo', 'juros', 'conta', 'agência', 'salário'],
  assento: ['cadeira', 'sofá', 'mesa', 'praça', 'madeira', 'sentar'],
  rio: ['rio', 'areia', 'margem', 'barco', 'praia', 'lago'],
};

const est = {
  toks: {},
  tok: null,
  worker: null,
  cfg: null,
  pronto: false,
  ids: [],
  nOriginal: 0,
  pass: null, // { estados, atencao, logits, T, ms }
  linhaSel: null,
  rng: null,
  historico: [],
  refs: null, // vetores de "banco" nas frases de referência, por camada
  ultimoSorteio: null,
};

// ---------- worker ----------
let seq = 0;
const pendentes = new Map();
function chamar(msg, transfer = []) {
  return new Promise((res, rej) => {
    const id = ++seq;
    pendentes.set(id, { res, rej });
    est.worker.postMessage({ ...msg, id }, transfer);
  });
}
function iniciarWorker() {
  est.worker = new Worker(new URL('./modelo.worker.js', import.meta.url), { type: 'module' });
  est.worker.onmessage = ({ data }) => {
    if (data.tipo === 'progresso') {
      $('cargaBarra').style.width = `${(100 * data.recebidos) / data.total}%`;
      $('cargaDetalhe').textContent = `${fmt(data.recebidos / 1e6, 1)} de ${fmt(data.total / 1e6, 1)} MB`;
      return;
    }
    if (data.tipo === 'etapa') { $('cargaDetalhe').textContent = data.texto; return; }
    const p = pendentes.get(data.id);
    if (!p) return;
    pendentes.delete(data.id);
    data.ok ? p.res(data.r) : p.rej(new Error(data.erro));
  };
}

async function carregarModelo() {
  $('carregar').disabled = true;
  $('carregar').textContent = 'Baixando…';
  if (!est.worker) iniciarWorker();
  try {
    const r = await chamar({ tipo: 'carregar', base: new URL('../modelo/', import.meta.url).href });
    est.cfg = r.cfg;
    est.pronto = true;
    $('cargaBarra').style.width = '100%';
    $('carregar').textContent = 'Modelo pronto';
    $('cargaTexto').innerHTML = `<b>Modelo pronto.</b> ${r.cfg.camadas} camadas, ${r.cfg.cabecas} cabeças de atenção por camada, vetores de ${r.cfg.dimensao} números e ${r.cfg.vocabulario.toLocaleString('pt-BR')} tokens no vocabulário.`;
    $('cargaDetalhe').textContent = `${r.doCache ? 'Veio do cache do navegador' : 'Baixado e guardado no cache'} · pronto em ${fmt(r.ms / 1000, 1)} s · WebAssembly, ${r.threads} thread${r.threads > 1 ? 's' : ''}`;
    try { localStorage.setItem('llm-modelo-ok', '1'); } catch { /* sem armazenamento */ }
    for (const b of ['passo', 'gerar10', 'recomecar']) $(b).disabled = false;
    await rodar();
  } catch (e) {
    $('carregar').disabled = false;
    $('carregar').textContent = 'Tentar de novo';
    $('cargaDetalhe').textContent = `Não deu para carregar: ${e.message}`;
  }
}

// ---------- tokenizadores ----------
async function tokenizador(id) {
  if (!est.toks[id]) est.toks[id] = AutoTokenizer.from_pretrained(id);
  return est.toks[id];
}
function brutos(tok, ids) {
  const v = tok.model.vocab;
  return ids.map((i) => (Array.isArray(v) ? v[i] : tok.model.convert_ids_to_tokens([i])[0]) ?? '');
}
function codificar(tok, texto) {
  return tok.encode(texto, { add_special_tokens: false });
}
function rotuloToken(id) {
  return textoDoToken(bytesDoToken(brutos(est.tok, [id])[0]));
}

function desenharTokens() {
  const box = $('listaTokens');
  box.innerHTML = '';
  if (!est.ids.length) { box.innerHTML = '<p class="vazio">Digite uma frase.</p>'; return; }
  const b = brutos(est.tok, est.ids);
  est.ids.forEach((id, i) => {
    const bytes = bytesDoToken(b[i]);
    const d = document.createElement('div');
    d.className = 'tok' + (i >= est.nOriginal ? ' novo' : '');
    d.innerHTML = `<b></b><span>ID ${id}</span><code></code>`;
    d.querySelector('b').textContent = textoDoToken(bytes);
    d.querySelector('code').textContent = hex(bytes);
    d.title = `posição ${i} · token bruto "${b[i]}"`;
    box.append(d);
  });
}

async function desenharComparacao() {
  const pt = $('frase').value;
  const en = $('fraseEn').value;
  const tbody = $('tabelaTok').querySelector('tbody');
  const linhas = [];
  for (const t of TOKENIZADORES) {
    let tok;
    try { tok = await tokenizador(t.id); } catch { continue; }
    const cel = (texto) => {
      const ids = codificar(tok, texto);
      const pedacos = brutos(tok, ids).map((s) => `<span>${esc(textoDoToken(bytesDoToken(s)))}</span>`).join('');
      return `<b>${ids.length} tokens</b><div class="mini" style="margin-top:4px">${pedacos}</div>`;
    };
    linhas.push(`<tr><td>${t.rotulo}</td><td>${cel(pt)}</td><td>${cel(en)}</td></tr>`);
  }
  tbody.innerHTML = linhas.join('');
}

function esc(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}

// ---------- rodar o modelo ----------
async function rodar() {
  desenharTokens();
  desenharContexto();
  if (!est.pronto || !est.ids.length) return;
  est.pass = await chamar({ tipo: 'passar', ids: est.ids });
  if (est.linhaSel == null || est.linhaSel >= est.pass.T) est.linhaSel = est.pass.T - 1;
  desenharAtencao();
  desenharProximo();
  await desenharVetores();
}

function estado(camada, pos) {
  const { estados, T } = est.pass;
  const D = est.cfg.dimensao;
  return estados.subarray((camada * T + pos) * D, (camada * T + pos + 1) * D);
}
function pesos(camada, cabeca, linha) {
  // camada 1..L, cabeca 0..H-1 ou -1 (média)
  const { atencao, T } = est.pass;
  const H = est.cfg.cabecas;
  const out = new Float32Array(T);
  const hs = cabeca < 0 ? [...Array(H).keys()] : [cabeca];
  for (const h of hs) {
    const o = (((camada - 1) * H + h) * T + linha) * T;
    for (let j = 0; j < T; j++) out[j] += atencao[o + j] / hs.length;
  }
  return out;
}

// ---------- etapa 2: vetores ----------
function vetoresReferencia() {
  if (!est.refs) est.refs = calcularReferencias();
  return est.refs;
}
async function calcularReferencias() {
  const tok = est.tok;
  const pontos = [];
  const total = Object.values(REFERENCIAS).flat().length;
  for (const [sentido, frases] of Object.entries(REFERENCIAS)) {
    for (const f of frases) {
      $('pcaBox').innerHTML = `<p class="vazio">Passando as ${total} frases de referência pelo modelo (${pontos.length + 1} de ${total})…</p>`;
      const ids = codificar(tok, f);
      const r = await chamar({ tipo: 'passar', ids, semLogits: true });
      const D = est.cfg.dimensao;
      const porCamada = [];
      for (let c = 0; c <= est.cfg.camadas; c++) porCamada.push(r.estados.slice((c * r.T + r.T - 1) * D, (c * r.T + r.T) * D));
      pontos.push({ sentido, frase: f, porCamada });
    }
  }
  return pontos;
}

function modoVetor() {
  return document.querySelector('#modoVetor button[aria-pressed="true"]').dataset.v;
}

async function desenharVetores() {
  if (!est.pronto || !est.pass) return;
  const box = $('pcaBox');
  const modo = modoVetor();
  $('grupoCamadaVet').style.display = modo === 'contexto' ? '' : 'none';
  let pontos;
  if (modo === 'contexto') {
    const camada = Number($('camadaVet').value);
    const refs = await vetoresReferencia();
    pontos = refs.map((r) => ({ v: r.porCamada[camada], cor: CORES[r.sentido], rotulo: '', dica: `“${r.frase}” (${r.sentido})`, raio: 6 }));
    // tokens "banco" da frase do usuário; se não houver, o último token
    const brutosFrase = brutos(est.tok, est.ids);
    let alvo = brutosFrase.map((b, i) => [b, i]).filter(([b]) => textoDoToken(bytesDoToken(b)).replace('␣', '').toLowerCase() === 'banco').map(([, i]) => i);
    if (!alvo.length) alvo = [est.ids.length - 1];
    const acc = acuraciaCentroide(refs.map((r) => r.porCamada[camada]), refs.map((r) => r.sentido));
    $('pcaMetrica').textContent = `Camada ${camada}: em ${pct(acc, 0)} das ${refs.length} frases, o vetor de "banco" fica mais perto do centroide do próprio sentido (medido nas ${est.cfg.dimensao} dimensões, não na projeção; acaso = 33%).`;
    for (const i of alvo) pontos.push({ v: estado(camada, i), cor: CORES.frase, rotulo: `sua frase: ${rotuloToken(est.ids[i])}`, dica: `“${rotuloToken(est.ids[i])}” na posição ${i} da sua frase`, raio: 7, destaque: true });
    $('pcaLegenda').innerHTML = Object.keys(REFERENCIAS).map((s) => `<span><i style="background:${CORES[s]}"></i>"banco" em frases de ${s === 'rio' ? 'rio (banco de areia)' : s}</span>`).join('') + `<span><i style="background:${CORES.frase}"></i>sua frase</span>`;
  } else {
    const grupos = [];
    const meta = [];
    const vistos = new Set();
    est.ids.forEach((id) => { if (!vistos.has(id)) { vistos.add(id); grupos.push([id]); meta.push({ cor: CORES.frase, rotulo: rotuloToken(id), dica: `token da sua frase (ID ${id})` }); } });
    for (const [g, palavras] of Object.entries(GRUPOS_ESTATICOS)) {
      for (const p of palavras) {
        const ids = codificar(est.tok, ' ' + p);
        grupos.push(ids);
        meta.push({ cor: CORES[g], rotulo: p, dica: `"${p}" (${ids.length} token${ids.length > 1 ? 's' : ''}; média dos vetores)` });
      }
    }
    const vs = await chamar({ tipo: 'embeddings', grupos });
    $('pcaMetrica').textContent = 'Vetores de entrada: a mesma linha da matriz de embeddings, qualquer que seja a frase. Palavras de mais de um token usam a média dos vetores.';
    pontos = vs.map((v, i) => ({ v, ...meta[i], raio: 5 }));
    $('pcaLegenda').innerHTML = `<span><i style="background:${CORES.frase}"></i>tokens da sua frase</span>` + Object.keys(GRUPOS_ESTATICOS).map((s) => `<span><i style="background:${CORES[s]}"></i>palavras de ${s === 'rio' ? 'rio' : s}</span>`).join('');
  }
  // No modo contexto, os eixos vêm só das frases de referência; a frase digitada é projetada neles.
  const nAjuste = modo === 'contexto' ? pontos.filter((p) => !p.destaque).length : pontos.length;
  const { coords, variancia } = pca2(pontos.map((p) => p.v), 120, nAjuste);
  $('pcaAviso').textContent = `É uma projeção: ${est.cfg.dimensao} dimensões espremidas em 2 (PCA). Os dois eixos guardam ${pct(variancia[0] + variancia[1], 0)} da variação desses pontos; pontos perto no plano podem estar longe no espaço original.`;
  dispersao(box, pontos, coords);
}

function dispersao(box, pontos, coords) {
  const L = 640, A = 360, m = 30;
  // escala pelos pontos de referência; pontos destacados muito longe ficam presos na borda (com aviso)
  const base = coords.filter((c, i) => !pontos[i].destaque);
  const xs = base.map((c) => c[0]), ys = base.map((c) => c[1]);
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const folga = 0.35;
  coords.forEach((c, i) => {
    if (!pontos[i].destaque) return;
    const [dx, dy] = [(x1 - x0) * folga, (y1 - y0) * folga];
    if (c[0] < x0 - dx || c[0] > x1 + dx || c[1] < y0 - dy || c[1] > y1 + dy) pontos[i].rotulo += ' (fora da escala)';
    x0 = Math.min(x0, Math.max(c[0], x0 - dx)); x1 = Math.max(x1, Math.min(c[0], x1 + dx));
    y0 = Math.min(y0, Math.max(c[1], y0 - dy)); y1 = Math.max(y1, Math.min(c[1], y1 + dy));
    c[0] = Math.min(Math.max(c[0], x0), x1); c[1] = Math.min(Math.max(c[1], y0), y1);
  });
  const sx = (x) => m + ((x - x0) / (x1 - x0 || 1)) * (L - 2 * m);
  const sy = (y) => A - m - ((y - y0) / (y1 - y0 || 1)) * (A - 2 * m);
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${L} ${A}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Projeção 2D dos vetores');
  svg.innerHTML = `<rect x="0" y="0" width="${L}" height="${A}" rx="10" fill="var(--fill)"/>`;
  const ordem = pontos.map((p, i) => i).sort((a, b) => (pontos[a].destaque ? 1 : 0) - (pontos[b].destaque ? 1 : 0));
  for (const i of ordem) {
    const p = pontos[i];
    const [x, y] = [sx(coords[i][0]), sy(coords[i][1])];
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', p.raio);
    c.setAttribute('fill', p.cor); c.setAttribute('stroke', 'var(--panel)'); c.setAttribute('stroke-width', '2');
    dica(c, p.dica);
    svg.append(c);
    if (p.rotulo) {
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('x', x + p.raio + 3); t.setAttribute('y', y + 4);
      if (p.destaque) t.setAttribute('class', 'forte');
      t.textContent = p.rotulo;
      svg.append(t);
    }
  }
  box.innerHTML = '';
  box.append(svg);
}

// ---------- etapa 3: atenção ----------
function desenharAtencao() {
  if (!est.pass) return;
  const camada = Number($('camadaAtt').value);
  const cabeca = Number($('cabeca').value);
  const T = est.pass.T;
  const rot = est.ids.map(rotuloToken);
  const cel = Math.max(14, Math.min(34, Math.floor(420 / T)));
  const margem = 96;
  const L = margem + T * cel + 8, A = margem + T * cel + 8;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${L} ${A}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `Mapa de atenção, camada ${camada}`);
  svg.innerHTML = `<defs><pattern id="hach" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--hachura)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--line)" stroke-width="1.5"/></pattern></defs>`;
  const fs = Math.min(11, cel * 0.7);
  for (let i = 0; i < T; i++) {
    const w = pesos(camada, cabeca, i);
    const tl = document.createElementNS(NS, 'text');
    tl.setAttribute('x', margem - 6); tl.setAttribute('y', margem + i * cel + cel / 2 + 4);
    tl.setAttribute('text-anchor', 'end'); tl.setAttribute('font-size', fs);
    if (i === est.linhaSel) tl.setAttribute('class', 'forte');
    tl.textContent = curto(rot[i]);
    svg.append(tl);
    for (let j = 0; j < T; j++) {
      const r = document.createElementNS(NS, 'rect');
      r.setAttribute('x', margem + j * cel); r.setAttribute('y', margem + i * cel);
      r.setAttribute('width', cel - 1); r.setAttribute('height', cel - 1); r.setAttribute('rx', 2);
      if (j > i) {
        r.setAttribute('fill', 'url(#hach)');
        dica(r, `máscara causal: “${rot[i]}” não pode ver “${rot[j]}”, que vem depois`);
      } else {
        r.setAttribute('fill', corMapa(w[j]));
        dica(r, `“${rot[i]}” → “${rot[j]}”: ${pct(w[j])}`);
      }
      r.style.cursor = 'pointer';
      r.addEventListener('click', () => { est.linhaSel = i; desenharAtencao(); });
      svg.append(r);
    }
  }
  for (let j = 0; j < T; j++) {
    const t = document.createElementNS(NS, 'text');
    const x = margem + j * cel + cel / 2 + 3, y = margem - 6;
    t.setAttribute('transform', `translate(${x} ${y}) rotate(-60)`);
    t.setAttribute('font-size', fs);
    t.textContent = curto(rot[j]);
    svg.append(t);
  }
  const sel = document.createElementNS(NS, 'rect');
  sel.setAttribute('x', margem - 2); sel.setAttribute('y', margem + est.linhaSel * cel - 1);
  sel.setAttribute('width', T * cel + 3); sel.setAttribute('height', cel + 1); sel.setAttribute('rx', 3);
  sel.setAttribute('fill', 'none'); sel.setAttribute('stroke', 'var(--accent)'); sel.setAttribute('stroke-width', 2);
  svg.append(sel);
  $('mapaBox').innerHTML = '';
  $('mapaBox').append(svg);
  desenharSetas(camada, cabeca, rot);
}

function curto(s) { return s.length > 12 ? s.slice(0, 11) + '…' : s; }

function corMapa(w) {
  // escala sequencial de um só tom (azul): 0 = fundo, 1 = azul forte
  const a = Math.pow(Math.min(1, Math.max(0, w)), 0.6);
  return `color-mix(in oklab, var(--accent) ${Math.round(a * 100)}%, var(--mapa-0))`;
}

function desenharSetas(camada, cabeca, rot) {
  const i = est.linhaSel;
  const w = pesos(camada, cabeca, i);
  const n = i + 1;
  const L = 320, linhaH = 24, A = Math.max(120, n * linhaH + 40);
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${L} ${A}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `Para onde “${rot[i]}” olha`);
  svg.innerHTML = `<defs><marker id="ponta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--accent)"/></marker></defs>`;
  const xQ = L - 70, yQ = A / 2, xK = 118;
  const t0 = document.createElementNS(NS, 'text');
  t0.setAttribute('x', 0); t0.setAttribute('y', 14); t0.setAttribute('class', 'muted');
  t0.textContent = `Para onde “${rot[i]}” olha (camada ${camada}${cabeca < 0 ? ', média das cabeças' : `, cabeça ${cabeca + 1}`})`;
  svg.append(t0);
  const ordem = [...w.slice(0, n)].map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]);
  const top = new Set(ordem.slice(0, 3).map((x) => x[1]));
  for (let j = 0; j < n; j++) {
    const y = 34 + j * linhaH + linhaH / 2;
    const t = document.createElementNS(NS, 'text');
    t.setAttribute('x', xK - 8); t.setAttribute('y', y + 4); t.setAttribute('text-anchor', 'end');
    if (top.has(j)) t.setAttribute('class', 'forte');
    t.textContent = top.has(j) ? `${curto(rot[j])} · ${pct(w[j], 0)}` : curto(rot[j]);
    svg.append(t);
    const p = document.createElementNS(NS, 'path');
    const espessura = 0.5 + 9 * w[j];
    p.setAttribute('d', `M${xQ} ${yQ} C ${(xQ + xK) / 2} ${yQ}, ${(xQ + xK) / 2} ${y}, ${xK} ${y}`);
    p.setAttribute('fill', 'none'); p.setAttribute('stroke', 'var(--accent)');
    p.setAttribute('stroke-width', espessura.toFixed(2)); p.setAttribute('stroke-opacity', (0.15 + 0.85 * Math.sqrt(w[j])).toFixed(2));
    if (w[j] > 0.05) p.setAttribute('marker-end', 'url(#ponta)');
    dica(p, `“${rot[i]}” → “${rot[j]}”: ${pct(w[j])}`);
    svg.append(p);
  }
  const tq = document.createElementNS(NS, 'text');
  tq.setAttribute('x', xQ + 8); tq.setAttribute('y', yQ + 4); tq.setAttribute('class', 'forte');
  tq.textContent = curto(rot[i]);
  svg.append(tq);
  $('setasBox').innerHTML = '';
  $('setasBox').append(svg);
}

// ---------- etapa 4: próximo token ----------
function opcoes() {
  return { temperatura: Number($('temp').value), topK: Number($('topk').value), topP: Number($('topp').value), limite: 12 };
}

function desenharProximo() {
  if (!est.pass) return;
  const o = opcoes();
  const d = distribuicao(est.pass.logits, o);
  const mantidos = new Set(d.mantidos);
  const lg = est.pass.logits;
  const max = d.probs[d.ordem[0]];
  const linhas = [`<div class="cand cab"><span>token</span><span>probabilidade (após temperatura)</span><span class="v">prob.</span><span class="l">logit</span></div>`];
  for (const id of d.ordem.slice(0, 12)) {
    const p = d.probs[id];
    const classe = ['cand', mantidos.has(id) ? '' : 'cortado'].join(' ');
    linhas.push(`<div class="${classe}" data-id="${id}" title="${mantidos.has(id) ? 'pode ser sorteado' : 'cortado pelo top-k ou top-p'}"><span class="t">${esc(rotuloToken(id))}</span><span class="b"><i style="width:${(100 * p) / max}%"></i></span><span class="v">${pct(p)}</span><span class="l">${fmt(lg[id], 1)}</span></div>`);
  }
  $('candidatos').innerHTML = linhas.join('');
  let massa = 0;
  for (const id of d.mantidos) massa += d.probs[id];
  let H = 0;
  d.mantidos.forEach((id, j) => { const p = d.finais[j]; if (p > 0) H -= p * Math.log2(p); });
  $('vMantidos').textContent = d.mantidos.length.toLocaleString('pt-BR');
  $('vMassa').textContent = pct(massa);
  $('vEntropia').textContent = fmt(H, 2);
  $('vRede').textContent = `${Math.round(est.pass.ms.rede + est.pass.ms.logits)} ms`;
  $('vRede').title = `rede: ${Math.round(est.pass.ms.rede)} ms · logits (50.257 produtos escalares): ${Math.round(est.pass.ms.logits)} ms`;
  desenharRoleta(d, false);
}

function desenharRoleta(d, comAgulha) {
  const box = $('roleta');
  box.innerHTML = '';
  const mostrar = d.mantidos.slice(0, 30);
  mostrar.forEach((id, j) => {
    const s = document.createElement('span');
    s.style.width = `${100 * d.finais[j]}%`;
    s.textContent = d.finais[j] > 0.06 ? rotuloToken(id) : '';
    if (comAgulha && est.ultimoSorteio?.id === id) s.className = 'sim';
    s.title = `${rotuloToken(id)}: ${pct(d.finais[j])}`;
    box.append(s);
  });
  if (comAgulha && est.ultimoSorteio) {
    const a = document.createElement('i');
    a.className = 'agulha';
    a.style.left = `${100 * est.ultimoSorteio.u}%`;
    box.append(a);
  }
  if (est.ultimoSorteio) {
    const T = opcoes().temperatura;
    $('explicaSorteio').textContent = T <= 0
      ? `Último passo: temperatura 0, sem sorteio; sai sempre o mais provável (“${rotuloToken(est.ultimoSorteio.id)}”).`
      : `Último sorteio: o número ${fmt(est.ultimoSorteio.u, 3)} caiu em “${rotuloToken(est.ultimoSorteio.id)}”, que tinha ${pct(est.ultimoSorteio.p)} depois dos cortes. A faixa abaixo já é a do próximo passo.`;
  }
}

function desenharContexto() {
  const box = $('contexto');
  box.innerHTML = '';
  est.ids.forEach((id, i) => {
    const s = document.createElement('span');
    s.textContent = rotuloToken(id);
    if (i >= est.nOriginal) s.className = 'gerado' + (i === est.ids.length - 1 ? ' ultimo' : '');
    box.append(s);
  });
  $('desfazer').disabled = !est.pronto || est.ids.length <= est.nOriginal;
}

async function passo(pausa = 700) {
  if (!est.pass || est.ids.length >= MAX_TOKENS) return;
  if (!est.rng) est.rng = mulberry32(Number($('semente').value) >>> 0);
  const o = opcoes();
  const d = distribuicao(est.pass.logits, o);
  let id, u = 0;
  if (o.temperatura <= 0) id = d.mantidos[0];
  else {
    // o mesmo sortear() de amostragem.js, guardando o número sorteado para desenhar a agulha
    id = sortear(d, () => { u = est.rng(); return u; });
  }
  est.historico.push(est.ids.length);
  est.ultimoSorteio = { id, u, p: d.finais[d.mantidos.indexOf(id)] };
  // mostra a agulha na distribuição que foi sorteada; depois o token entra no contexto e tudo recalcula
  desenharRoleta(d, true);
  marcarEscolhido(id);
  await new Promise((r) => setTimeout(r, pausa));
  est.ids = [...est.ids, id];
  est.linhaSel = est.ids.length - 1;
  await rodar();
}

function marcarEscolhido(id) {
  document.querySelectorAll('#candidatos .cand[data-id]').forEach((el) => el.classList.toggle('escolhido', Number(el.dataset.id) === id));
}

let gerando = false;
async function gerar(n) {
  if (gerando) return;
  gerando = true;
  $('gerar10').disabled = true;
  try { for (let k = 0; k < n; k++) await passo(250); } finally { gerando = false; $('gerar10').disabled = false; }
}

// ---------- tooltip ----------
function dica(el, texto) {
  el.addEventListener('pointerenter', (e) => mostrarDica(e, texto));
  el.addEventListener('pointermove', (e) => mostrarDica(e, texto));
  el.addEventListener('pointerleave', () => { $('dica').style.display = 'none'; });
}
function mostrarDica(e, texto) {
  const d = $('dica');
  d.textContent = texto;
  d.style.display = 'block';
  d.style.left = `${Math.min(window.innerWidth - 120, Math.max(120, e.clientX))}px`;
  d.style.top = `${e.clientY + window.scrollY}px`;
}

// ---------- resultados dos experimentos ----------
async function desenharExperimentos() {
  const pega = (n) => fetch(`resultados/${n}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const [banco, tokens, temp, atr] = await Promise.all(['banco', 'tokens', 'temperatura', 'atribuicao'].map(pega));
  const cards = [];
  if (banco) {
    const c1 = banco.por_camada[1], c0 = banco.por_camada[0], ult = banco.por_camada[banco.por_camada.length - 1];
    cards.push(['banco_camadas.png', '1. Os sentidos de "banco" se separam nas camadas',
      `Em ${banco.n_frases} frases (financeiro, assento e banco de areia), o vetor de "banco" acerta o sentido pelo centroide mais próximo em ${pct(c0.acuracia_loo, 0)} das vezes na entrada (só a posição muda), ${pct(c1.acuracia_loo, 0)} já depois da camada 1 e ${pct(ult.acuracia_loo, 0)} na última. E em "O banco fica perto do rio?" o vetor de "banco" é idêntico ao de "O banco aprovou o empréstimo.": a máscara causal impede que ele veja o rio.`]);
  }
  if (tokens) {
    const t = Object.fromEntries(tokens.tokenizadores.map((x) => [x.nome, x]));
    cards.push(['tokens_pt_en.png', '2. Português custa mais tokens (depende do tokenizador)',
      `No WMT24++ (${tokens.segmentos} trechos com tradução humana), o mesmo conteúdo em português usa ${fmt(t['GPT-2 (inglês)'].razao_pt_en)}× os tokens do inglês no GPT-2 original, ${fmt(t['Qwen2.5'].razao_pt_en)}× no Qwen2.5 e ${fmt(t['o200k (GPT-4o)'].razao_pt_en)}× no tokenizador do GPT-4o. No GPT-2 português a conta se inverte: ${fmt(t['GPT-2 português'].razao_pt_en)}×.`]);
  }
  if (temp) {
    const l = (p, T) => temp.linhas.find((x) => x.prompt === temp.prompts[p] && x.temperatura === T);
    cards.push(['temperatura.png', '3. Temperatura troca repetição por surpresa',
      `${temp.sementes} sementes por temperatura, ${temp.novos_tokens} tokens cada. Em "${temp.prompts[1]}…", temperatura 0 repete ${pct(l(1, 0).repeticao_interna, 0)} dos trigramas; a 0,8 a repetição some, e a 1,5 o texto vira sopa: a surpresa média sobe de ${fmt(l(1, 0).surpresa_media, 1)} para ${fmt(l(1, 1.5).surpresa_media, 1)} nats por token.`]);
  }
  if (atr) {
    const s = atr.atencao_no_1o_token_por_camada;
    cards.push(['atencao_x_atribuicao.png', '4. Atenção não é explicação',
      `Em ${atr.n_frases} frases, a atenção concorda pouco com a oclusão (Spearman médio entre ${fmt(Math.min(...atr.spearman_atencao_por_camada))} e ${fmt(Math.max(...atr.spearman_atencao_por_camada))} por camada), e o token que mais recebe atenção na última camada nunca foi o mais importante (${pct(atr.top1_igual_oclusao.atencao_ultima_camada, 0)}). Motivo principal: ${pct(s[s.length - 1], 0)} da atenção vai para o primeiro token, que pesa só ${pct(atr.oclusao_peso_relativo_1o_token, 0)} na oclusão.`]);
  }
  $('figuras').innerHTML = cards.map(([img, t, p]) => `<article class="card"><h3>${t}</h3><p>${p}</p><img src="figuras/${img}" alt="${esc(t)}" loading="lazy"></article>`).join('');
}

// ---------- eventos ----------
async function analisar() {
  if (!est.tok) return;
  est.ids = codificar(est.tok, $('frase').value.trim() || ' ').slice(0, MAX_TOKENS - 10);
  est.nOriginal = est.ids.length;
  est.historico = [];
  est.linhaSel = null;
  est.rng = null;
  est.ultimoSorteio = null;
  $('explicaSorteio').textContent = 'Os tokens que sobraram dividem a faixa de 0 a 1 conforme a probabilidade. Um número da semente cai em um deles.';
  desenharComparacao();
  await rodar();
}

function ligarFaixa(id, saida, f, depois) {
  const atualizar = () => { $(saida).textContent = f(Number($(id).value)); depois?.(); };
  $(id).addEventListener('input', atualizar);
  atualizar();
}

async function iniciar() {
  const sel = $('cabeca');
  sel.innerHTML = '<option value="-1">Média das 12</option>' + Array.from({ length: 12 }, (_, h) => `<option value="${h}">Cabeça ${h + 1}</option>`).join('');
  ligarFaixa('temp', 'tempV', (v) => fmt(v), () => desenharProximo());
  ligarFaixa('topk', 'topkV', (v) => (v === 0 ? 'off' : String(v)), () => desenharProximo());
  ligarFaixa('topp', 'toppV', (v) => fmt(v), () => desenharProximo());
  ligarFaixa('camadaAtt', 'camadaAttV', (v) => String(v), () => desenharAtencao());
  ligarFaixa('camadaVet', 'camadaVetV', (v) => (v === 0 ? '0 (entrada)' : String(v)), () => desenharVetores());
  sel.addEventListener('change', desenharAtencao);
  $('semente').addEventListener('change', () => { est.rng = null; });
  $('analisar').addEventListener('click', analisar);
  $('frase').addEventListener('keydown', (e) => { if (e.key === 'Enter') analisar(); });
  $('fraseEn').addEventListener('input', () => desenharComparacao());
  $('sugestoes').addEventListener('click', (e) => { if (e.target.matches('.chip')) { $('frase').value = e.target.textContent; analisar(); } });
  $('carregar').addEventListener('click', carregarModelo);
  $('passo').addEventListener('click', () => passo());
  $('gerar10').addEventListener('click', () => gerar(10));
  $('recomecar').addEventListener('click', analisar);
  $('desfazer').addEventListener('click', async () => {
    const n = est.historico.pop();
    if (n == null) return;
    est.ids = est.ids.slice(0, n);
    est.ultimoSorteio = null;
    est.linhaSel = est.ids.length - 1;
    await rodar();
  });
  document.querySelectorAll('#modoVetor button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('#modoVetor button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    desenharVetores();
  }));
  desenharExperimentos();
  try {
    est.tok = await tokenizador('gpt2-pt');
  } catch (e) {
    $('listaTokens').innerHTML = `<p class="vazio">Não deu para carregar o tokenizador: ${esc(e.message)}</p>`;
    return;
  }
  await analisar();
  // quem já baixou o modelo antes (está no cache) não precisa clicar de novo
  let jaBaixou = false;
  try { jaBaixou = localStorage.getItem('llm-modelo-ok') === '1'; } catch { /* sem armazenamento */ }
  if (jaBaixou) carregarModelo();
}

iniciar();
