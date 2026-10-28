"""Packaging-Guards fuer den root-basierten Worker-Death-Stack."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines()}


def _pos(text: str, needle: str) -> int:
    pos = text.find(needle)
    assert pos >= 0, f"{needle!r} nicht gefunden"
    return pos


def _last_pos(text: str, needle: str) -> int:
    pos = text.rfind(needle)
    assert pos >= 0, f"{needle!r} nicht gefunden"
    return pos


def _shell_function(text: str, name: str) -> str:
    start = _pos(text, f"{name}() {{")
    end = text.find("\n}\n\n", start)
    assert end >= 0, f"Ende von {name!r} nicht gefunden"
    return text[start : end + 3]


def test_postinst_keeps_root_executed_code_root_owned():
    postinst = _read("debian/postinst")
    lines = _lines(postinst)

    assert "chown -R root:root /opt/kiara" in postinst
    assert "chmod -R u=rwX,go=rX /opt/kiara" in postinst
    assert "chown -R kiara:kiara /opt/kiara/.cache" in postinst
    assert "chown -R kiara:kiara /opt/kiara" not in lines


def test_deploy_keeps_root_executed_code_root_owned():
    deploy = _read("deploy.sh")
    lines = _lines(deploy)

    assert "chown -R root:root /opt/kiara" in deploy
    assert "chmod -R u=rwX,go=rX /opt/kiara" in deploy
    assert "chown -R kiara:kiara /opt/kiara/.cache" in deploy
    assert "chown -R kiara:kiara /opt/kiara" not in lines


def test_postinst_and_deploy_protect_worker_death_state_directory():
    for relpath in ("debian/postinst", "deploy.sh"):
        script = _read(relpath)
        lines = _lines(script)
        assert "chown root:kiara /var/lib/kiara" in script
        assert "chmod 1775 /var/lib/kiara" in script
        assert "chown -R root:root /var/lib/kiara/worker_death_watcher" in script
        assert "chmod 0700 /var/lib/kiara/worker_death_watcher" in script
        assert "chown -R kiara:kiara /var/lib/kiara" not in lines


def test_worker_death_watcher_unit_only_writes_own_state_dir():
    unit = _read("debian/kiara-worker-death-watcher.service")

    assert "User=root" in unit
    assert "ExecStart=/opt/kiara/venv/bin/python3 -u -m kiara.web.worker_death_watcher" in unit
    assert "ReadWritePaths=/var/lib/kiara/worker_death_watcher /var/log/kiara" in unit
    assert "ReadWritePaths=/var/lib/kiara /var/log/kiara" not in unit


def test_cifs_flight_recorder_is_a_hardened_oneshot_before_worker():
    unit = _read("debian/kiara-cifs-flight-recorder.service")
    worker = _read("debian/kiara-worker.service")
    rules = _read("debian/rules")
    postinst = _read("debian/postinst")
    deploy = _read("deploy.sh")

    assert "Type=oneshot" in unit
    assert "ExecStartPre=/usr/sbin/modprobe cifs" in unit
    assert "ExecStart=/opt/kiara/venv/bin/kiara prepare-cifs-flight-recorder" in unit
    assert "CapabilityBoundingSet=CAP_SYS_MODULE" in unit
    assert "ReadOnlyPaths=/sys" in unit
    assert "ReadWritePaths=/sys/kernel/tracing" in unit
    assert "Before=kiara-worker.service" in unit
    assert "Wants=kiara-cifs-flight-recorder.service" in worker
    assert "After=network.target kiara-cifs-flight-recorder.service" in worker
    assert "install -m 644 debian/kiara-cifs-flight-recorder.service" in rules
    assert "systemctl enable kiara-cifs-flight-recorder.service" in deploy
    assert "kiara-cifs-flight-recorder.service" in postinst


def test_smb_pcap_recorder_is_bounded_hardened_and_starts_before_worker():
    unit = _read("debian/kiara-smb-pcap-recorder.service")
    worker = _read("debian/kiara-worker.service")
    rules = _read("debian/rules")
    postinst = _read("debian/postinst")
    deploy = _read("deploy.sh")
    control = _read("debian/control")

    assert "ExecStart=/usr/bin/tcpdump" in unit
    assert "User=tcpdump" in unit
    assert "-Z root" not in unit
    assert "-s 256" in unit
    assert "-C 8 -W 8" in unit
    assert "StateDirectory=kiara-smb-forensics" in unit
    assert "CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_RAW" in unit
    assert "ProtectSystem=strict" in unit
    assert "Before=kiara-worker.service" in unit
    assert "Wants=kiara-cifs-flight-recorder.service kiara-smb-pcap-recorder.service" in worker
    assert "kiara-smb-pcap-recorder.service" in postinst
    assert "systemctl enable kiara-smb-pcap-recorder.service" in deploy
    assert "install -m 644 debian/kiara-smb-pcap-recorder.service" in rules
    assert "tcpdump" in control


def test_privileged_stall_collector_sudoers_is_narrow_and_versioned():
    expected = (
        "kiara ALL=(root) NOPASSWD: /opt/kiara/venv/bin/kiara "
        "^collect-ingestion-stall-forensics [0-9]{1,7} [0-9]{1,7}$"
    )
    for relpath in ("debian/postinst", "deploy.sh"):
        script = _read(relpath)
        assert expected in script
        assert "chmod 0440 /etc/sudoers.d/kiara-ingestion-forensics" in script


def test_deploy_restarts_worker_death_watcher_unit():
    deploy = _read("deploy.sh")

    assert "kiara-worker-death-watcher.service" in deploy
    assert 'systemctl restart "$svc"' in deploy


def test_deploy_installs_systemd_units_before_restart():
    deploy = _read("deploy.sh")

    assert "install -d /usr/lib/systemd/system" in deploy
    assert "for unit in debian/kiara*.service; do" in deploy
    assert 'install -m 0644 "$unit" /usr/lib/systemd/system/' in deploy
    assert "systemctl daemon-reload" in deploy


def test_deploy_rebuilds_trusted_venv_before_final_venv_execution():
    deploy = _read("deploy.sh")

    assert "/opt/kiara/venv/bin/pip" not in deploy
    assert "VENV_BUILD=/opt/kiara/venv.build.$$" in deploy
    assert 'python3 -m venv "$VENV_BUILD"' in deploy
    assert 'PIP="$VENV_BUILD/bin/pip"' in deploy
    assert "rewrite_venv_shebangs()" in deploy
    assert 'sed -i "1s|^#!$VENV_BUILD/bin/python' in deploy
    assert "systemctl stop kiara-worker-death-watcher.service" in deploy
    assert "protect_opt_tree_without_venv()" in deploy
    assert _last_pos(deploy, "protect_opt_tree_without_venv") < _pos(
        deploy, 'python3 -m venv "$VENV_BUILD"'
    )
    assert _pos(deploy, "systemctl stop kiara-worker-death-watcher.service") < _pos(
        deploy, 'python3 -m venv "$VENV_BUILD"'
    )
    assert _pos(deploy, 'python3 -m venv "$VENV_BUILD"') < _pos(
        deploy, 'mv "$VENV_BUILD" "$VENV_FINAL"'
    )
    assert _pos(deploy, "stoppe Services fuer venv-Swap") < _pos(
        deploy, 'mv "$VENV_BUILD" "$VENV_FINAL"'
    )
    assert _pos(deploy, 'mv "$VENV_BUILD" "$VENV_FINAL"') < _last_pos(
        deploy, "rewrite_venv_shebangs"
    )
    assert _pos(deploy, 'mv "$VENV_BUILD" "$VENV_FINAL"') < _pos(deploy, "[4/4] Restart Services")


def test_deploy_fast_path_uses_final_pip_only_after_trust_gate_decision():
    deploy = _read("deploy.sh")

    decision_block = deploy[
        _pos(deploy, 'rebuild_reason=""') : _pos(deploy, 'if [ -n "$rebuild_reason" ]; then')
    ]
    assert 'elif ! venv_trust_gate "$new_fingerprint"; then' in decision_block
    assert 'rebuild_reason="venv Trust-Gate nicht bestanden"' in decision_block
    assert "else\n    fast_update_local_packages\nfi" in deploy
    assert _pos(deploy, 'elif ! venv_trust_gate "$new_fingerprint"; then') < _last_pos(
        deploy, "fast_update_local_packages"
    )
    assert 'PIP="$VENV_FINAL/bin/pip"' in deploy


def test_deploy_trust_gate_checks_root_owned_and_not_group_writable_venv():
    deploy = _read("deploy.sh")

    assert 'if [ ! -d "$VENV_FINAL" ] || [ -L "$VENV_FINAL" ]; then' in deploy
    assert 'find "$VENV_FINAL" \\( ! -user root -o ! -group root \\) -print -quit' in deploy
    assert 'find "$VENV_FINAL" \\( -type f -o -type d \\) -perm /022 -print -quit' in deploy
    assert 'if [ ! -f "$VENV_FINGERPRINT" ] || [ -L "$VENV_FINGERPRINT" ]; then' in deploy
    assert "stat -c '%u:%g' \"$VENV_FINGERPRINT\"" in deploy
    assert 'find "$VENV_FINGERPRINT" -perm /022 -print -quit' in deploy
    assert 'rebuild_reason="venv Trust-Gate nicht bestanden"' in deploy
    assert "elif venv_final_path_invalid; then" in deploy
    assert 'rebuild_reason="venv ist Symlink oder kein echtes Verzeichnis"' in deploy


def test_deploy_missing_or_invalid_trust_marker_forces_full_rebuild():
    deploy = _read("deploy.sh")

    assert "VENV_TRUST_DIR=/var/lib/kiara/deploy" in deploy
    assert 'VENV_TRUST_MARKER="$VENV_TRUST_DIR/venv.trusted"' in deploy
    assert 'ensure_root_dir_no_symlink "$VENV_TRUST_DIR" 0700' in deploy
    assert 'if [ ! -f "$VENV_TRUST_MARKER" ]; then' in deploy
    assert "Trust-Marker fehlt" in deploy
    assert "stat -c '%u:%g' \"$VENV_TRUST_MARKER\"" in deploy
    assert 'find "$VENV_TRUST_MARKER" -perm /022 -print -quit' in deploy
    assert 'grep -Fxq "venv=$VENV_FINAL" "$VENV_TRUST_MARKER"' in deploy
    assert "marker_fingerprint=$(awk -F=" in deploy
    assert 'if [ "$marker_fingerprint" != "$expected_fingerprint" ]; then' in deploy
    assert _pos(deploy, 'venv_trust_marker_valid "$expected_fingerprint"') < _pos(
        deploy, 'rebuild_reason="venv Trust-Gate nicht bestanden"'
    )


def test_deploy_writes_root_only_trust_marker_after_full_rebuild():
    deploy = _read("deploy.sh")

    rebuild_block = deploy[
        _pos(deploy, 'if [ -n "$rebuild_reason" ]; then') : _pos(
            deploy, "else\n    fast_update_local_packages\nfi"
        )
    ]
    assert 'build_trusted_venv "$new_fingerprint"' in rebuild_block
    assert "swap_in_built_venv" in rebuild_block
    assert 'mark_venv_trusted "$new_fingerprint"' in rebuild_block
    assert _pos(rebuild_block, 'build_trusted_venv "$new_fingerprint"') < _pos(
        rebuild_block, "swap_in_built_venv"
    )
    assert _pos(rebuild_block, "swap_in_built_venv") < _pos(
        rebuild_block, 'mark_venv_trusted "$new_fingerprint"'
    )
    assert 'ensure_root_dir_no_symlink "$VENV_TRUST_DIR" 0700' in deploy
    assert 'chown root:root "$marker_tmp"' in deploy
    assert 'chmod 0600 "$marker_tmp"' in deploy
    assert 'mv -f "$marker_tmp" "$VENV_TRUST_MARKER"' in deploy


def test_deploy_does_not_sanitize_existing_venv_before_trust_gate():
    deploy = _read("deploy.sh")
    pre_gate = deploy[: _pos(deploy, 'elif ! venv_trust_gate "$new_fingerprint"; then')]

    assert "protect_opt_tree_without_venv" in pre_gate
    assert '-path "$VENV_FINAL" -prune' in pre_gate
    assert "-exec chown -h root:root {} +" in pre_gate
    assert "\\( -type f -o -type d \\)" in pre_gate
    assert _last_pos(pre_gate, "protect_opt_tree_without_venv") < _pos(
        pre_gate, "new_fingerprint=$(dependency_fingerprint)"
    )
    assert "chown -R root:root /opt/kiara" not in pre_gate
    assert "chmod -R u=rwX,go=rX /opt/kiara" not in pre_gate


def test_deploy_trust_and_pip_cache_dirs_are_symlink_hardened():
    deploy = _read("deploy.sh")

    assert "ensure_root_dir_no_symlink() {" in deploy
    assert '[ -L "$dir" ] || { [ -e "$dir" ] && [ ! -d "$dir" ]; }' in deploy
    assert 'rm -rf "$dir"' in deploy
    assert 'install -d -o root -g root -m "$mode" "$dir"' in deploy
    assert 'ensure_root_dir_no_symlink "$VENV_TRUST_DIR" 0700' in deploy
    assert "ensure_root_dir_no_symlink /var/cache/kiara/pip 0755" in deploy
    assert "install -d -m 0755 /var/cache/kiara/pip" not in deploy


def test_deploy_reads_existing_fingerprint_bounded_from_regular_file_only():
    deploy = _read("deploy.sh")
    reader = _shell_function(deploy, "read_existing_fingerprint")

    assert 'if [ ! -d "$VENV_FINAL" ] || [ -L "$VENV_FINAL" ]; then' in reader
    assert 'if [ ! -f "$VENV_FINGERPRINT" ] || [ -L "$VENV_FINGERPRINT" ]; then' in reader
    assert 'head -c 65 "$VENV_FINGERPRINT"' in reader
    assert "grep -Eq '^[0-9a-f]{64}$'" in reader
    assert "old_fingerprint=$(read_existing_fingerprint)" in deploy
    assert 'old_fingerprint=$(cat "$VENV_FINGERPRINT")' not in deploy


def test_ensure_root_dir_no_symlink_replaces_directory_symlink(tmp_path):
    if os.geteuid() != 0:
        pytest.skip("install -d -o root requires root")

    deploy = _read("deploy.sh")
    function = _shell_function(deploy, "ensure_root_dir_no_symlink")
    target = tmp_path / "target"
    link = tmp_path / "link"
    target.mkdir()
    (target / "sentinel").write_text("kept")
    link.symlink_to(target, target_is_directory=True)

    subprocess.run(
        [
            "bash",
            "-euc",
            "\n".join(
                [
                    function,
                    f"ensure_root_dir_no_symlink {shlex.quote(str(link))} 0700",
                    f"[ -d {shlex.quote(str(link))} ]",
                    f"[ ! -L {shlex.quote(str(link))} ]",
                    f"[ -f {shlex.quote(str(target / 'sentinel'))} ]",
                ]
            ),
        ],
        check=True,
    )


def test_postinst_rebuilds_trusted_venv_before_final_venv_execution():
    postinst = _read("debian/postinst")

    assert "/opt/kiara/venv/bin/pip" not in postinst
    assert "VENV_BUILD=/opt/kiara/venv.build.$$" in postinst
    assert 'python3 -m venv "$VENV_BUILD"' in postinst
    assert 'PIP="$VENV_BUILD/bin/pip"' in postinst
    assert "rewrite_venv_shebangs()" in postinst
    assert 'sed -i "1s|^#!$VENV_BUILD/bin/python' in postinst
    assert "systemctl stop kiara-worker-death-watcher.service" in postinst
    assert _pos(postinst, "chown -R root:root /opt/kiara") < _pos(
        postinst, 'python3 -m venv "$VENV_BUILD"'
    )
    assert _pos(postinst, "systemctl stop kiara-worker-death-watcher.service") < _pos(
        postinst, 'python3 -m venv "$VENV_BUILD"'
    )
    assert _pos(postinst, 'python3 -m venv "$VENV_BUILD"') < _pos(
        postinst, 'mv "$VENV_BUILD" "$VENV_FINAL"'
    )
    assert _pos(postinst, "kiara-worker-death-watcher.service; do") < _pos(
        postinst, 'mv "$VENV_BUILD" "$VENV_FINAL"'
    )
    assert _pos(postinst, 'mv "$VENV_BUILD" "$VENV_FINAL"') < _last_pos(
        postinst, "rewrite_venv_shebangs"
    )
    assert _pos(postinst, 'mv "$VENV_BUILD" "$VENV_FINAL"') < _pos(
        postinst, "/opt/kiara/venv/bin/kiara init-db"
    )
