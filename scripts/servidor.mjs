// Servidor estático mínimo para a pasta web/ (sem dependências).
// Uso: node scripts/servidor.mjs [porta] [--isolado]
// --isolado envia COOP/COEP para liberar WASM com várias threads.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const raiz = resolve(fileURLToPath(new URL('../web', import.meta.url)));
const porta = Number(process.argv[2] || 8765);
const isolado = process.argv.includes('--isolado');
const tipos = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.onnx': 'application/octet-stream', '.wasm': 'application/wasm',
};

http.createServer(async (req, res) => {
  const caminho = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
  let arquivo = join(raiz, caminho);
  if (!arquivo.startsWith(raiz)) { res.writeHead(403); return res.end(); }
  if (caminho.endsWith('/') || caminho === '\\') arquivo = join(arquivo, 'index.html');
  try {
    const corpo = await readFile(arquivo);
    const cab = { 'Content-Type': tipos[extname(arquivo)] || 'application/octet-stream', 'Cache-Control': 'no-store' };
    if (isolado) Object.assign(cab, { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' });
    res.writeHead(200, cab);
    res.end(corpo);
  } catch {
    res.writeHead(404);
    res.end('não encontrado');
  }
}).listen(porta, () => console.log(`servindo ${raiz} em http://localhost:${porta}`));
