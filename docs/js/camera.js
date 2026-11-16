// Camera: lens picker, zoom, torch, resolution.
const $ = id => document.getElementById(id);
export const cam = { stream: null, track: null, list: [], id: null };
try { cam.id = localStorage.getItem('set.camId'); } catch {}

function lensLabel(c, i) {
  const l = c.label.toLowerCase();
  if (/ultra/.test(l)) return '0.5×';
  if (/tele/.test(l)) return 'Tele';
  if (/triple|dual/.test(l)) return 'Auto';
  if (/^(back camera|rückkamera|camera achter)$/i.test(c.label.trim())) return '1×';
  return `Lens ${i + 1}`;
}

async function listCameras(onError) {
  const all = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const back = all.filter(d => !/front|user|face|voor/i.test(d.label));
  cam.list = back.length ? back : all;
  const sel = $('camSelect'); sel.innerHTML = '';
  all.forEach(d => { const o = document.createElement('option'); o.value = d.deviceId; o.textContent = d.label || ('camera ' + d.deviceId.slice(0, 6)); sel.appendChild(o); });
  const lb = $('lensButtons'); lb.innerHTML = '';
  if (cam.list.length > 1) cam.list.forEach((c, i) => {
    const b = document.createElement('button'); b.textContent = lensLabel(c, i); b.title = c.label; b.dataset.id = c.deviceId;
    b.onclick = () => openCamera(c.deviceId).catch(onError); lb.appendChild(b);
  });
}

function markActive() {
  const id = cam.track && cam.track.getSettings().deviceId;
  $('camSelect').value = id || '';
  for (const b of $('lensButtons').children) b.classList.toggle('on', b.dataset.id === id);
}

let onOpen = () => {};
export async function openCamera(deviceId) {
  const video = $('video');
  if (cam.stream) cam.stream.getTracks().forEach(t => t.stop());
  const h = +$('camRes').value, base = { width: { ideal: Math.round(h * 16 / 9) }, height: { ideal: h }, frameRate: { ideal: 30 } };
  const c = deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode: { ideal: 'environment' } };
  cam.stream = await navigator.mediaDevices.getUserMedia({ video: c, audio: false });
  cam.track = cam.stream.getVideoTracks()[0];
  video.srcObject = cam.stream;
  await video.play().catch(() => {});
  cam.id = cam.track.getSettings().deviceId;
  try { localStorage.setItem('set.camId', cam.id); } catch {}
  setupTrackControls(); markActive(); onOpen();
}

function setupTrackControls() {
  const tr = cam.track, caps = tr.getCapabilities ? tr.getCapabilities() : {};
  if (caps.focusMode && caps.focusMode.includes('continuous')) tr.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }).catch(() => {});
  $('torchBtn').style.display = caps.torch ? '' : 'none';
  $('torchBtn').classList.remove('on');
  const zr = $('zoomRow'), z = $('zoom');
  if (caps.zoom && caps.zoom.max > caps.zoom.min) {
    zr.style.display = 'flex';
    z.min = caps.zoom.min; z.max = Math.min(caps.zoom.max, 4); z.step = caps.zoom.step || 0.1;
    const cur = tr.getSettings().zoom ?? 1, start = caps.zoom.min < 1 ? caps.zoom.min : cur;
    z.value = start; $('zoomVal').textContent = (+start).toFixed(1) + '×';
    if (start !== cur) tr.applyConstraints({ advanced: [{ zoom: start }] }).catch(() => {});
  } else zr.style.display = 'none';
  const s = tr.getSettings(); $('camInfo').textContent = `${s.width}×${s.height}`;
}

export async function startCamera(onError, onOpened) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API not available. Open this page over https:// (or localhost).');
  onOpen = onOpened || onOpen;
  await openCamera(cam.id);
  await listCameras(onError);
  let chosen = null; try { chosen = localStorage.getItem('set.chosenOnce'); } catch {}
  if (!chosen) {
    const uw = cam.list.find(c => /ultra/i.test(c.label));
    if (uw && uw.deviceId !== cam.id) await openCamera(uw.deviceId);
    try { localStorage.setItem('set.chosenOnce', '1'); } catch {}
  }
  markActive();
}

export function setupCameraControls(onError) {
  $('zoom').addEventListener('input', e => {
    const v = +e.target.value; $('zoomVal').textContent = v.toFixed(1) + '×';
    cam.track?.applyConstraints({ advanced: [{ zoom: v }] }).catch(() => {});
  });
  let torchOn = false;
  $('torchBtn').addEventListener('click', async () => {
    if (!cam.track) return;
    torchOn = !torchOn;
    try { await cam.track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch { torchOn = false; onError(new Error('Flash not available on this lens')); }
    $('torchBtn').classList.toggle('on', torchOn);
  });
  $('camSelect').addEventListener('change', e => openCamera(e.target.value).catch(onError));
  $('camRes').addEventListener('change', () => cam.stream && openCamera(cam.id).catch(onError));
}
