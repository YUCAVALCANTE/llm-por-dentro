import test from 'node:test';
import assert from 'node:assert/strict';
import { embeddingDe, logitsDeEstado, pca2, acuraciaCentroide, cosseno } from '../src/vetores.js';
import { bytesDoToken, hex, textoDoToken } from '../src/bytes.js';

test('embedding int8 com escala por linha e logits por produto escalar', () => {
  const D = 3;
  const q = Int8Array.from([1, 2, 3, -1, 0, 1]);
  const esc = Float32Array.from([0.5, 2]);
  assert.deepEqual(Array.from(embeddingDe(q, esc, D, 1)), [-2, 0, 2]);
  const h = Float32Array.from([1, 1, 1]);
  assert.deepEqual(Array.from(logitsDeEstado(q, esc, D, h)), [3, 0]);
});

test('PCA acha o eixo de maior variância e é determinístico', () => {
  const X = [];
  for (let i = -5; i <= 5; i++) X.push(Float32Array.from([i, 0.1 * (i % 2), 0]));
  const a = pca2(X), b = pca2(X);
  assert.deepEqual(a.coords, b.coords);
  assert.ok(a.variancia[0] > 0.99);
  assert.ok(Math.abs(Math.abs(a.coords[10][0]) - 5) < 1e-2);
});

test('acurácia do centroide: grupos separados dão 100%', () => {
  const X = [[1, 0], [1.1, 0.1], [0.9, -0.1], [-1, 0], [-1.1, 0.1], [-0.9, -0.1]].map((v) => Float32Array.from(v));
  assert.equal(acuraciaCentroide(X, ['a', 'a', 'a', 'b', 'b', 'b']), 1);
  assert.ok(Math.abs(cosseno([1, 0], [0, 1])) < 1e-12);
});

test('bytes de tokens BPE em nível de byte', () => {
  assert.equal(hex(bytesDoToken('Ġbanco')), '20 62 61 6e 63 6f');
  // "ã" = c3 a3; no vocabulário do GPT-2 os dois bytes viram "Ã£"
  assert.equal(hex(bytesDoToken('Ã£')), 'c3 a3');
  assert.equal(textoDoToken(bytesDoToken('ĠnÃ£o')), '␣não');
  // token que termina no meio de um caractere
  assert.equal(textoDoToken(Uint8Array.from([0x6e, 0xc3])), 'n‹c3›');
});
