"""Fuehrt die produktive Warnungsanzeige inklusive Chat-Reload in Node aus."""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
SOURCE = Path(__file__).resolve().parents[1] / "kiara/web/static/js/chat.js"


@pytest.mark.skipif(not NODE, reason="Node is required for frontend tests")
def test_warning_updates_in_place_and_restores_with_its_message():
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const escape = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const ctx = {
    document: {readyState: 'loading', addEventListener() {}},
    SharedUi: {
        runtime: {escapeHtml: escape, polling: {start() {}}},
        markdown: {render: s => s},
    },
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
const ChatUI = vm.runInContext('ChatUI', ctx);
function makeMessage() {
    const text = {innerHTML: ''};
    const bubble = {querySelector: sel => sel === '.chat-text' ? text : null};
    return {dataset: {}, blocks: new Map(), querySelector: sel => sel === '.chat-bubble' ? bubble : null};
}
const rendered = [];
let appendCount = 0;
const ui = Object.create(ChatUI.prototype);
ui._view = {
    getCustomBlockEl(id, {target}) { return target.blocks.get(id); },
    appendCustomBlock(html, {target, id}) {
        appendCount++;
        target.blocks.set(id, {innerHTML: html, remove() { target.blocks.delete(id); }});
    },
    startAssistantMessage() { const el = makeMessage(); rendered.push(el); return el; },
    setSources() {}, scrollToBottom() {},
};
const message = makeMessage();
const warning = {code: 'reranking_unavailable', message: 'Relevanzsortierung ausgefallen.', count: 1};
ui._renderProcessingWarnings(message, [warning]);
assert.equal(message.blocks.size, 1);
assert.match(message.blocks.get('processing-warnings').innerHTML, /Eingeschränkte Suchqualität/);
const oldBlock = message.blocks.get('processing-warnings');
ui._renderProcessingWarnings(message, [{...warning, count: 7}]);
assert.equal(message.blocks.get('processing-warnings'), oldBlock);
assert.equal(appendCount, 1);
assert.match(oldBlock.innerHTML, /7× aufgetreten/);
assert.match(oldBlock.innerHTML, /role="status"/);
const second = makeMessage();
ui._renderProcessingWarnings(second, []);
assert.equal(second.blocks.size, 0);
assert.equal(message.blocks.size, 1);
ui._renderProcessingWarnings(second, [{...warning, message: '<img src=x onerror=alert(1)>', count: 'x<img>'}]);
assert.ok(!second.blocks.get('processing-warnings').innerHTML.includes('<img'));
assert.match(second.blocks.get('processing-warnings').innerHTML, /&lt;img/);
ui._renderMessages([
    {role: 'assistant', content: 'Antwort mit Hinweis', warnings: [{...warning, count: 7}]},
    {role: 'assistant', content: 'Gesunde Antwort', warnings: []},
]);
assert.equal(rendered.length, 2);
assert.match(rendered[0].blocks.get('processing-warnings').innerHTML, /7× aufgetreten/);
assert.equal(rendered[1].blocks.size, 0);
assert.equal(rendered[0].querySelector('.chat-bubble').querySelector('.chat-text').innerHTML, 'Antwort mit Hinweis');
"""
    result = subprocess.run(
        [NODE, "-e", script, str(SOURCE)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
