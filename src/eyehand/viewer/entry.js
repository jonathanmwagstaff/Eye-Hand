const input = document.querySelector('#patient-name');
const panel = document.querySelector('#entry-panel');
const status = document.querySelector('#voice-status');
const preview = document.querySelector('#voice-preview');
const enable = document.querySelector('#enable-microphone');
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition, mode = null, silence, restart, active = false, wanted = false, blocked = false, committed = '', audio;
const characters = {
  ay: 'A', aye: 'A', bee: 'B', be: 'B', see: 'C', sea: 'C', cee: 'C', dee: 'D',
  ee: 'E', eff: 'F', gee: 'G', aitch: 'H', haitch: 'H', eye: 'I', jay: 'J', kay: 'K',
  el: 'L', ell: 'L', em: 'M', en: 'N', oh: 'O', pee: 'P', pea: 'P', cue: 'Q', queue: 'Q',
  are: 'R', ar: 'R', ess: 'S', tee: 'T', tea: 'T', you: 'U', vee: 'V',
  ex: 'X', why: 'Y', zed: 'Z', zee: 'Z', space: ' ', hyphen: '-', dash: '-', apostrophe: "'", period: '.', dot: '.',
  zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9'
};
function spell(text) {
  text = text.toLowerCase().replace(/\b([a-z]) as in [a-z]+\b/g, '$1').replace(/double[ -](?:you|u)/g, 'w');
  const tokens = text.match(/[a-z]+|[0-9]|['.-]/g) || [];
  let value = '', unknown = [];
  for (const token of tokens) {
    if (characters[token] !== undefined) value += characters[token];
    else if (/^[a-z0-9'.-]$/.test(token)) value += token.toUpperCase();
    else unknown.push(token);
  }
  return {value, unknown};
}
function unlockAudio() {
  try {
    audio ||= new (window.AudioContext || window.webkitAudioContext)();
    audio.resume().catch(() => {});
  } catch { /* Audio output may be unavailable. */ }
}
function chime(start) {
  try {
    unlockAudio();
    const time = audio.currentTime;
    for (const [i, frequency] of (start ? [523.25, 783.99] : [783.99, 523.25]).entries()) {
      const oscillator = audio.createOscillator(), gain = audio.createGain(), at = time + i * 0.16;
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0, at);
      gain.gain.linearRampToValueAtTime(0.13, at + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.001, at + 0.14);
      oscillator.connect(gain); gain.connect(audio.destination);
      oscillator.start(at); oscillator.stop(at + 0.15);
    }
  } catch { /* Speech remains usable when audio output is unavailable. */ }
}
function finish() {
  clearTimeout(silence);
  if (mode) chime(false);
  mode = null;
  input.removeAttribute('data-mode');
  status.textContent = 'Ready — say “Write this” or “Spell this”.';
}
function armSilence() { clearTimeout(silence); if (mode) silence = setTimeout(finish, 3000); }
function append(text, final) {
  let value;
  if (mode === 'spell') {
    const result = spell(text);
    value = committed + result.value;
    preview.textContent = result.unknown.length ? `Not entered: ${result.unknown.join(', ')}. Try the letter name, or “A as in apple”.` : '';
  } else {
    value = committed + (committed && !/\s$/.test(committed) && text ? ' ' : '') + text;
  }
  input.value = value;
  if (final) committed = value;
}
function process(text, final) {
  const command = text.match(/^(write this|type this|spell this)\b[\s,:.!-]*(.*)$/i);
  if (command) {
    // Wait for a final command so interim revisions cannot repeatedly change modes.
    if (!final) return;
    mode = /^spell/i.test(command[1]) ? 'spell' : 'write';
    committed = input.value;
    input.setAttribute('data-mode', mode);
    status.textContent = mode === 'spell' ? 'Spelling — listening…' : 'Dictating — listening…';
    preview.textContent = '';
    chime(true);
    if (command[2]) append(command[2], true);
    armSilence();
    return;
  }
  if (!mode) return;
  if (/^done writing[.!]?$/i.test(text.trim())) { if (final) finish(); return; }
  if (/^delete last character[.!]?$/i.test(text.trim())) {
    if (final) { committed = committed.slice(0, -1); input.value = committed; }
  } else append(text, final);
  armSilence();
}
function selected() { return !panel.hidden && document.activeElement === input && !document.hidden && document.hasFocus(); }
function stop() {
  clearTimeout(restart);
  finish();
  active = false;
  wanted = false;
  recognition?.abort();
  status.textContent = 'Select Patient Name to begin.';
}
window.nameEntry = {stop};
function start() {
  if (!Recognition) { status.textContent = 'Browser speech recognition is unavailable. Use Chrome or Edge.'; return; }
  if (active || blocked || !selected()) return;
  recognition ||= new Recognition();
  recognition.lang = 'en-US';
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.onstart = () => { status.textContent = 'Ready — say “Write this” or “Spell this”.'; };
  recognition.onresult = event => {
    if (!active || blocked || !selected()) return;
    for (let i = event.resultIndex; i < event.results.length; i++) {
      process(event.results[i][0].transcript.trim(), event.results[i].isFinal);
    }
  };
  recognition.onspeechstart = () => clearTimeout(silence);
  recognition.onspeechend = armSilence;
  recognition.onerror = event => {
    if (event.error === 'aborted' || event.error === 'no-speech') return;
    blocked = true;
    finish();
    status.textContent = event.error === 'not-allowed' || event.error === 'service-not-allowed'
      ? 'Microphone or speech access denied. Allow access in the browser, then click Enable microphone.'
      : `Speech recognition error: ${event.error}. Click Enable microphone to retry.`;
  };
  recognition.onend = () => {
    active = false;
    if (wanted && !blocked && selected()) restart = setTimeout(start, 400);
  };
  try { active = wanted = true; recognition.start(); }
  catch (error) { active = false; status.textContent = `Could not start speech recognition: ${error.message}`; }
}
async function permission() {
  if (!Recognition) { start(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    for (const track of stream.getTracks()) track.stop();
    blocked = false;
    if (selected()) start();
  } catch {
    blocked = true;
    status.textContent = 'Allow microphone access in the browser, then click Enable microphone.';
  }
}
input.addEventListener('click', () => { unlockAudio(); input.focus(); start(); });
input.addEventListener('focus', () => { committed = input.value; start(); });
input.addEventListener('blur', stop);
input.addEventListener('input', () => { committed = input.value; });
enable.addEventListener('click', async () => {
  unlockAudio();
  await permission();
  input.focus(); start();
});
document.querySelector('#clear-name').addEventListener('click', () => {
  finish(); input.value = committed = ''; preview.textContent = ''; input.focus(); start();
});
window.addEventListener('blur', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
permission();
