"""Speech results, focus and silence handling without a live microphone."""
from pathlib import Path
import shutil
import subprocess

import pytest
from eyehand import browser_cursor


def test_voice_name_entry(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for speech UI check')
    script = r'''
const vm = require('node:vm'), fs = require('node:fs'), assert = require('node:assert/strict');
(async () => {
const elements = {}, timers = new Map(), frequencies = [], windowEvents = {};
let speech, audioStopped = false, permissionRequests = 0, nextTimer = 0;
const document = {activeElement: null, hidden: false, hasFocus: () => true,
  addEventListener() {}, querySelector(id) {
    return elements[id] ||= {value: '', hidden: id === '#entry-panel', attrs: {}, listeners: {},
      setAttribute(k,v) {this.attrs[k]=v;}, removeAttribute(k) {delete this.attrs[k];},
      addEventListener(k,v) {this.listeners[k]=v;},
      focus() {document.activeElement=this;this.listeners.focus?.();}
    };
  }};
class Recognition {
  constructor() {speech=this;}
  start() {this.starts=(this.starts||0)+1;this.onstart?.();}
  abort() {this.aborted=true;this.onend?.();}
}
class AudioContext {
  currentTime=0; destination={};
  resume() {return Promise.resolve();}
  createOscillator() {return {frequency: {set value(v) {frequencies.push(v);}},connect(){},start(){},stop(){}};}
  createGain() {return {gain: {setValueAtTime(){},linearRampToValueAtTime(){},exponentialRampToValueAtTime(){}},connect(){}};}
}
const scope = {document, window: {SpeechRecognition: Recognition, AudioContext,
  addEventListener(k,v){windowEvents[k]=v;}},
  navigator: {mediaDevices: {async getUserMedia() {permissionRequests++; return {getTracks: () => [{stop() {audioStopped=true;}}]};}}},
  setTimeout(fn, ms) {const id=++nextTimer;timers.set(id,{fn,ms});return id;}, clearTimeout(id){timers.delete(id);},console};
vm.runInNewContext(fs.readFileSync('ENTRY_PATH','utf8'),scope);
await new Promise(setImmediate);
assert.equal(permissionRequests,1);assert.equal(audioStopped,true);
assert.equal(speech,undefined); // Permission request does not start transcription in another tab.
const input=elements['#patient-name'];elements['#entry-panel'].hidden=false;
input.listeners.click();
assert.equal(speech.starts,1);
function say(text, final=true) {
 const result=[{transcript:text}];result.isFinal=final;
 speech.onresult({resultIndex:0,results:[result]});
}
say('unrelated conversation');assert.equal(input.value,'');
say('write this Jordan');assert.equal(input.value,'Jordan');
assert.equal(input.attrs['data-mode'],'write');
assert.deepEqual(frequencies.slice(-2),[523.25,783.99]);
say('Smi',false);assert.equal(input.value,'Jordan Smi');
say('Smith',false);assert.equal(input.value,'Jordan Smith');
say('Smith');assert.equal(input.value,'Jordan Smith');
const timeout=[...timers.values()].find(t=>t.ms===3000);assert.ok(timeout);timeout.fn();
assert.equal(input.attrs['data-mode'],undefined);
assert.deepEqual(frequencies.slice(-2),[783.99,523.25]);
say('ignore this');assert.equal(input.value,'Jordan Smith');
elements['#clear-name'].listeners.click();assert.equal(input.value,'');
say('spell this jay oh are dee ay en space ess em eye tee aitch');
assert.equal(input.value,'JORDAN SMITH');
say('space hyphen apostrophe period zero two');assert.equal(input.value,"JORDAN SMITH -'.02");
say('delete last character');assert.equal(input.value,"JORDAN SMITH -'.0");
say('done writing');assert.equal(input.attrs['data-mode'],undefined);
elements['#clear-name'].listeners.click();
say('spell this A as in apple space B as in bravo');assert.equal(input.value,'A B');
say('confusing');assert.equal(input.value,'A B');assert.match(elements['#voice-preview'].textContent,/confusing/);
scope.window.nameEntry.stop();assert.equal(speech.aborted,true);
say('write this forbidden');assert.equal(input.value,'A B'); // Stop must also gate late results.
input.listeners.click();
speech.onerror({error:'not-allowed'});speech.onend();
assert.match(elements['#voice-status'].textContent,/denied/);
assert.ok(![...timers.values()].some(t=>t.ms===400));
})();
'''
    script = script.replace('ENTRY_PATH', str(Path(browser_cursor.__file__).parent / 'viewer' / 'entry.js'))
    path = tmp_path / 'check.cjs'
    path.write_text(script)
    result = subprocess.run([node, str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
