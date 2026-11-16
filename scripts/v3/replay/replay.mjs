// Replay a game video through the real web app (docs/) in headless Chromium, with the video as the
// camera, and log every processing step (window.__last) for scripts/v3/score_replay.py.
//
//   cd scripts/v3/replay && npm install
//   ffmpeg -i data/videos/game_hq.mp4 -an -q:v 4 -pix_fmt yuvj420p scratch/game_hq.mjpeg   (Chromium wants MJPEG/Y4M)
//   node scripts/v3/replay/replay.mjs docs scratch/game_hq.mjpeg scratch/replay.json 76 '{"set.camRes":"1080"}'
//
// Notes: headless Chromium has no WebGPU adapter here, so models run on single-threaded WASM, which is
// much slower than a phone. Chromium's fake camera plays the file in real time, so a slow app sees fewer frames.
import { chromium } from 'playwright-core';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const [docs, video, out, secs = '75', ls = '{}'] = process.argv.slice(2);
if (!out) { console.error('usage: node replay.mjs <docsDir> <video.mjpeg> <out.json> [seconds] [localStorage JSON]'); process.exit(1); }
const types = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.css': 'text/css' };
const srv = http.createServer((req, res) => {
  let p = path.join(docs, decodeURIComponent(req.url.split('?')[0]));
  if (p.endsWith('/')) p += 'index.html';
  fs.readFile(p, (e, d) => { if (e) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' }); res.end(d); });
}).listen(0);

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || '/usr/bin/chromium', headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${path.resolve(video)}`,
    '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on('pageerror', e => console.log('[pageerror]', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('[page]', m.text().slice(0, 200)); });
await page.addInitScript(lsJson => {
  for (const [k, v] of Object.entries(JSON.parse(lsJson))) localStorage.setItem(k, v);
  try { delete Navigator.prototype.gpu; } catch {}   // no WebGPU adapter headless; skip the slow failed attempt
  window.__log = []; window.__camOpen = 0;
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async c => { const s = await gum(c); window.__camOpen = performance.now(); return s; };
  let last;
  Object.defineProperty(window, '__last', {
    get: () => last,
    set: v => {
      last = v;
      const cards = (v.cards || []).map(c => [c.name, c.cert ?? null, c.seen ?? true, c.box]);
      window.__log.push({ ...v, cards, t: performance.now() - window.__camOpen, status: document.getElementById('status')?.textContent || '' });
    },
  });
}, ls);
await page.goto(`http://localhost:${srv.address().port}/`, { waitUntil: 'commit' });
await page.waitForFunction(() => /loaded/i.test(document.getElementById('loadMsg')?.textContent || ''), null, { timeout: 180000 });
await page.click('#introBtn');
await page.waitForTimeout(+secs * 1000);
const log = await page.evaluate(() => window.__log);
fs.writeFileSync(out, JSON.stringify(log));
console.log('steps', log.length);
await browser.close(); srv.close();
