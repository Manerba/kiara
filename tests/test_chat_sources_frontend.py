"""Exercise source controls and mode changes with the actual JavaScript."""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
SOURCE = Path(__file__).resolve().parents[1] / "kiara/web/static/js/tab_chat.js"
pytestmark = pytest.mark.skipif(not NODE, reason="Node is required for frontend tests")

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const stored = new Map();
const elements = {};
for (const id of ['chatFilterK', 'chatFilterKSlider', 'chatSourceAuto', 'chatSourceAutoGroup']) {
    elements[id] = {value: '', max: '', checked: false, disabled: false,
        classList: {toggle(name, hidden) { this.hidden = hidden; }}};
}
const ctx = {
    document: {getElementById: id => elements[id]},
    sessionStorage: {getItem: key => stored.get(key), setItem: (key, value) => stored.set(key, value)},
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
const api = vm.runInContext(`({
    apply: _applyChatSourceProfile, get: _getChatSourceK, remember: _rememberChatSourceChoice,
    profiles: value => _chatSourceProfiles = value, clear: () => _chatSourceChoices.clear(),
})`, ctx);
const profiles = {
    standard: {default_k: 12, max_k: 80},
    rewriter: {default_k: 32, max_k: 100},
    agent: {default_k: 20, max_k: 100, auto_enabled: true, auto_default_k: 18},
};
api.profiles(profiles);
const input = elements.chatFilterK, auto = elements.chatSourceAuto;
"""


def _run(script):
    result = subprocess.run(
        [NODE, "-e", HARNESS + script, str(SOURCE)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mode_defaults_manual_choice_and_session_restore():
    _run(r"""
api.apply('standard');
assert.equal(api.get('standard'), 12);
input.value = '50'; api.remember();
api.apply('agent');
assert.equal(api.get('agent'), 20);
input.value = '60'; api.remember();
api.apply('rewriter');
assert.equal(api.get('rewriter'), 32);
api.apply('standard');
assert.equal(api.get('standard'), 50, 'switching modes preserves UI setting');
api.clear();
api.apply('agent');
assert.equal(api.get('agent'), 60, 'reload restores saved setting');
profiles.agent.max_k = 40;
api.apply('agent');
assert.equal(api.get('agent'), 20, 'a reduced admin limit replaces invalid saved values');
""")


def test_auto_zero_without_fallback_and_disabled_auto():
    _run(r"""
profiles.agent.default_k = 0;
api.apply('agent');
assert.equal(api.get('agent'), 0);
assert.equal(input.disabled, true);
assert.equal(input.value, 18);
api.remember();
api.apply('standard');
assert.equal(api.get('standard'), 12);
assert.equal(auto.checked, false);
api.apply('agent');
assert.equal(api.get('agent'), 0, 'Auto must not become a numeric fallback');
profiles.agent.auto_enabled = false;
profiles.agent.default_k = 20;
api.apply('agent');
assert.equal(auto.checked, false);
assert.equal(input.disabled, false);
assert.equal(api.get('agent'), 18);
assert.equal(elements.chatSourceAutoGroup.classList.hidden, true);
assert.equal(api.get('unchained'), null);
""")


def test_invalid_input_or_missing_profiles_prevent_submission():
    _run(r"""
api.apply('standard');
for (const invalid of ['', '0', '2025', '2.5', '-1', 'summe']) {
    input.value = invalid;
    assert.throws(() => api.get('standard'), /ganze Zahl/);
}
auto.checked = true;
assert.throws(() => api.get('standard'), /nicht freigegeben/);
api.profiles({});
api.apply('agent');
assert.equal(input.disabled, true);
assert.throws(() => api.get('agent'), /neu laden/);
""")
