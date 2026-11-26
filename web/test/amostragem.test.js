import test from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, softmax, distribuicao, proximoToken } from '../src/amostragem.js';

const LOGITS = [2.0, 1.0, 0.5, 0.5, -1.0];

test('mulberry32 é determinístico e igual ao Python', () => {
  const r = mulberry32(42);
  // mesmos valores em python/tests/test_amostragem.py
  assert.deepEqual([r(), r(), r()], [0.6011037519201636, 0.44829055899754167, 0.8524657934904099]);
});

test('softmax soma 1, é estável e respeita a ordem', () => {
  const p = softmax([1000, 999, 0]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(p[0] > p[1] && p[1] > p[2]);
});

test('temperatura 0 vira escolha gulosa; temperatura alta achata', () => {
  assert.deepEqual(Array.from(softmax(LOGITS, 0)), [1, 0, 0, 0, 0]);
  const quente = softmax(LOGITS, 100);
  assert.ok(Math.max(...quente) - Math.min(...quente) < 0.01);
});

test('top-k e top-p cortam e renormalizam', () => {
  const d = distribuicao(LOGITS, { temperatura: 0.7, topK: 3, topP: 0.9 });
  assert.deepEqual(d.mantidos, [0, 1, 2]);
  assert.ok(Math.abs(d.finais.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(Math.abs(d.finais[0] - 0.736935857641613) < 1e-12);
  const k1 = distribuicao(LOGITS, { topK: 1 });
  assert.deepEqual(k1.mantidos, [0]);
  const p = distribuicao(LOGITS, { topP: 0.01 });
  assert.equal(p.mantidos.length, 1);
});

test('empates: menor ID primeiro', () => {
  const d = distribuicao([0, 3, 3, 1], { topK: 2 });
  assert.deepEqual(d.mantidos, [1, 2]);
});

test('amostragem com semente reproduz a sequência do Python', () => {
  const g = mulberry32(7);
  const seq = Array.from({ length: 10 }, () => proximoToken(LOGITS, { temperatura: 1.2 }, g));
  assert.deepEqual(seq, [0, 0, 4, 2, 1, 0, 0, 0, 1, 2]);
});

test('frequências da amostragem batem com as probabilidades', () => {
  const g = mulberry32(1);
  const n = 20000;
  const cont = new Array(5).fill(0);
  for (let i = 0; i < n; i++) cont[proximoToken(LOGITS, { temperatura: 1 }, g)]++;
  const p = softmax(LOGITS, 1);
  for (let i = 0; i < 5; i++) assert.ok(Math.abs(cont[i] / n - p[i]) < 0.015, `token ${i}`);
});
