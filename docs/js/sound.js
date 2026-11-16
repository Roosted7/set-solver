// "Set!" sound: voice, beep or the user's own recording.
const $ = id => document.getElementById(id);
let audioCtx = null, customAudio = null, lastAt = 0;

export function unlockAudio() {
  try { audioCtx ||= new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch {}
  if (customAudio) { customAudio.muted = true; customAudio.play().then(() => { customAudio.pause(); customAudio.muted = false; customAudio.currentTime = 0; }).catch(() => {}); }
  if ('speechSynthesis' in window) { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); }
}

function beep() {
  if (!audioCtx) return;
  const t = audioCtx.currentTime;
  [880, 1320].forEach((f, i) => {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.frequency.value = f; o.type = 'square';
    g.gain.setValueAtTime(0.0001, t + i * 0.12); g.gain.exponentialRampToValueAtTime(0.25, t + i * 0.12 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.12 + 0.18);
    o.connect(g).connect(audioCtx.destination); o.start(t + i * 0.12); o.stop(t + i * 0.12 + 0.2);
  });
}

export function playSound(mode) {
  const now = performance.now();
  if (mode === 'off' || now - lastAt < 1200) return;
  lastAt = now;
  if (mode === 'file' && customAudio) { customAudio.currentTime = 0; customAudio.play().catch(beep); return; }
  if (mode === 'voice' && 'speechSynthesis' in window) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance('Set!'); u.rate = 1.1; u.pitch = 1.2; speechSynthesis.speak(u); return;
  }
  beep();
}

export function setupSoundSettings() {
  const sync = () => { $('soundFileRow').style.display = $('soundMode').value === 'file' ? '' : 'none'; };
  $('soundMode').addEventListener('change', sync); sync();
  $('soundFile').addEventListener('change', e => { const f = e.target.files[0]; if (f) { customAudio = new Audio(URL.createObjectURL(f)); customAudio.preload = 'auto'; } });
}
