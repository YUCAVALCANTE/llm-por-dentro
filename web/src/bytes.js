// Tokens BPE "em nível de byte" (GPT-2, Qwen): cada byte vira um caractere imprimível no vocabulário
// (o espaço vira "Ġ", por exemplo). Aqui desfazemos esse mapa para mostrar os bytes de verdade.

function bytesParaUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const mapa = new Map();
  bs.forEach((b, i) => mapa.set(String.fromCharCode(cs[i]), b));
  return mapa;
}

const DECODIFICADOR = bytesParaUnicode();

// "Ġbanco" -> Uint8Array [0x20, 0x62, ...]. Caracteres fora do mapa (tokens especiais) vão em UTF-8.
export function bytesDoToken(tokenBruto) {
  const out = [];
  for (const ch of tokenBruto) {
    const b = DECODIFICADOR.get(ch);
    if (b !== undefined) out.push(b);
    else out.push(...new TextEncoder().encode(ch));
  }
  return Uint8Array.from(out);
}

export function hex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join(' ');
}

// Texto legível de um token: decodifica os bytes; se o token termina no meio de um caractere
// UTF-8 (acontece com acentos), mostra os bytes que sobram como ‹c3›.
export function textoDoToken(bytes) {
  const dec = new TextDecoder('utf-8', { fatal: false });
  let s = '';
  let i = 0;
  while (i < bytes.length) {
    const b = bytes[i];
    const n = b < 0x80 ? 1 : b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 0;
    if (n === 0 || i + n > bytes.length) { s += `‹${b.toString(16).padStart(2, '0')}›`; i++; continue; }
    const pedaco = bytes.subarray(i, i + n);
    let ok = true;
    for (let j = 1; j < n; j++) if ((pedaco[j] & 0xc0) !== 0x80) ok = false;
    if (!ok) { s += `‹${b.toString(16).padStart(2, '0')}›`; i++; continue; }
    s += dec.decode(pedaco);
    i += n;
  }
  return s.replace(/ /g, '␣').replace(/\n/g, '⏎');
}
