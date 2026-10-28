"""Tests fuer den privilegierten CIFS-D-State-Forensikpfad."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kiara.web import ingestion_forensics as forensics


def _make_trace_instance(trace_root: Path, trace: str = "") -> Path:
    instance = trace_root / "instances" / "kiara-cifs"
    instance.mkdir(parents=True)
    (instance / "tracing_on").write_text("1\n")
    (instance / "buffer_size_kb").write_text("64\n")
    (instance / "trace").write_text(trace)
    (instance / "trace_stat").write_text("entries: 2\n")
    return instance


def _make_kretprobe_controls(trace_root: Path, instance: Path, *, definition: str = ""):
    (trace_root / "kprobe_events").write_text(definition)
    (trace_root / "kprobe_profile").write_text(
        "  kiara_cifs/decrypt_raw_data_ret                         0               0\n"
    )
    event_dir = instance / "events" / "kiara_cifs" / "decrypt_raw_data_ret"
    event_dir.mkdir(parents=True)
    (event_dir / "enable").write_text("0\n")
    (event_dir / "filter").write_text("none\n")


def test_configure_cifs_flight_recorder_augments_without_losing_trace(tmp_path):
    instance = _make_trace_instance(tmp_path, "incident evidence\n")
    _make_kretprobe_controls(tmp_path, instance)
    for event in forensics._TRACE_EVENTS:
        event_dir = instance / "events" / "cifs" / event
        event_dir.mkdir(parents=True)
        (event_dir / "enable").write_text("0\n")

    result = forensics.configure_cifs_flight_recorder(trace_root=tmp_path)

    assert result["available"] is True
    assert result["buffer_kb_per_cpu"] == 256
    assert result["preserved_existing"] is True
    assert result["enabled_events"] == list(forensics._TRACE_EVENTS)
    assert (instance / "tracing_on").read_text() == "1\n"
    assert (instance / "buffer_size_kb").read_text() == "64\n"
    assert (instance / "trace").read_text() == "incident evidence\n"
    assert all(
        (instance / "events" / "cifs" / event / "enable").read_text() == "1\n"
        for event in forensics._TRACE_EVENTS
    )
    assert forensics._KRETPROBE_DEFINITION in (tmp_path / "kprobe_events").read_text()
    probe_dir = instance / "events" / "kiara_cifs" / "decrypt_raw_data_ret"
    assert (probe_dir / "filter").read_text() == "ret < 0\n"
    assert (probe_dir / "enable").read_text() == "1\n"
    assert result["kretprobe"]["created"] is True
    assert result["kretprobe"]["enabled"] is True

    repeated = forensics.configure_cifs_flight_recorder(trace_root=tmp_path)
    assert repeated["preserved_existing"] is True
    assert (instance / "trace").read_text() == "incident evidence\n"


def test_trace_snapshot_correlates_pending_mid_and_resumes_recorder(tmp_path):
    trace = (
        "kiara-51 [001] smb3_cmd_enter: sid=0x1 tid=0x2 cmd=8 mid=77\n"
        "cifsd-12 [002] smb3_hdr_credits: current_mid=77 credits=8 credit_change=1\n"
        "cifsd-12 [002] decrypt_raw_data_ret: ret=-74\n"
        "cifsd-12 [002] smb3_cmd_done: sid=0x1 tid=0x2 cmd=13 mid=76\n"
    )
    instance = _make_trace_instance(tmp_path, trace)
    _make_kretprobe_controls(
        tmp_path,
        instance,
        definition=f"{forensics._KRETPROBE_DEFINITION}\n",
    )

    result = forensics.read_cifs_flight_recorder(
        [{"state": 2, "command": 8, "pid": 51, "mid": 77}],
        trace_root=tmp_path,
    )

    assert result["available"] is True
    assert result["line_count"] == 4
    assert result["pending_mid_matches"] == trace.splitlines()[:2]
    assert result["pending_thread_matches"] == [trace.splitlines()[0]]
    assert result["decrypt_failures"] == [trace.splitlines()[2]]
    assert result["credit_events"] == [trace.splitlines()[1]]
    assert result["pending_mid_analysis"][0]["command_enter_seen"] is True
    assert result["pending_mid_analysis"][0]["command_done_seen"] is False
    assert result["pending_mid_analysis"][0]["command_error_seen"] is False
    assert "smb3_cmd_enter" in result["tail"]
    assert (instance / "tracing_on").read_text() == "1\n"


def _make_fake_worker_proc(
    proc_root: Path, *, cgroup: str = "0::/system.slice/kiara-worker.service"
):
    pid = 1234
    tid = 1235
    process_dir = proc_root / str(pid)
    task_dir = process_dir / "task" / str(tid)
    task_dir.mkdir(parents=True)
    (process_dir / "status").write_text("Name:\tkiara\nUid:\t999\t999\t999\t999\n")
    (process_dir / "cgroup").write_text(f"{cgroup}\n")
    (process_dir / "cmdline").write_bytes(b"/opt/kiara/venv/bin/kiara\0serve-worker\0")
    # Nach dem geklammerten comm beginnt Feld 3; Index 19 ist starttime (Feld 22).
    (process_dir / "stat").write_text(f"{pid} (kiara) {' '.join(['S', *(['0'] * 18), '555'])}\n")
    (process_dir / "mountinfo").write_text(
        "42 31 0:52 / /mnt/kiara-ds-32 rw,relatime - cifs //192.168.1.10/cloud rw,seal\n"
    )

    files = {
        "status": "Name:\tingestion-worker\nState:\tD (disk sleep)\n",
        "stat": f"{tid} (kiara) D 0 0 0\n",
        "wchan": "folio_wait_bit_common\n",
        "stack": "cifs_strict_readv\nfilemap_read\n",
        "syscall": "0 0x7 0xabc 0x72d 0x0 0x0 0x0 0xdef 0xbeef\n",
        "sched": "nr_switches : 99\n",
        "io": "rchar: 1837\n",
    }
    for name, raw in files.items():
        (task_dir / name).write_text(raw)

    target = proc_root / "fixture.md"
    target.write_text("fixture")
    (process_dir / "fd").mkdir()
    (process_dir / "fd" / "7").symlink_to(target)
    (process_dir / "fdinfo").mkdir()
    (process_dir / "fdinfo" / "7").write_text("pos:\t0\nflags:\t0100000\nmnt_id:\t42\n")

    cifs_root = proc_root / "fs" / "cifs"
    cifs_root.mkdir(parents=True)
    (cifs_root / "DebugData").write_text("MIDs:\n\tState: 2 com: 8 pid: 1235 cbdata: 0000 mid 77\n")
    (cifs_root / "Stats").write_text("Operations (MIDs): 1\n")
    (cifs_root / "open_files").write_text("fixture.md\n")
    boot_dir = proc_root / "sys" / "kernel" / "random"
    boot_dir.mkdir(parents=True)
    (boot_dir / "boot_id").write_text("test-boot-id\n")
    (proc_root / "uptime").write_text("123.0 456.0\n")
    return pid, tid


def test_collector_captures_syscall_fd_mid_mount_and_trace(tmp_path):
    proc_root = tmp_path / "proc"
    trace_root = tmp_path / "trace"
    pid, tid = _make_fake_worker_proc(proc_root)
    _make_trace_instance(
        trace_root,
        "kiara-1235 [001] smb3_cmd_enter: sid=0x1 tid=0x2 cmd=8 mid=77\n",
    )

    with (
        patch.object(forensics, "_capture_smb_sockets", return_value={"available": True}),
        patch.object(forensics, "_capture_relevant_dmesg", return_value={"available": True}),
        patch.object(forensics, "_capture_pcap_ring", return_value={"available": True}),
    ):
        result = forensics.collect_ingestion_stall_forensics(
            pid,
            tid,
            proc_root=proc_root,
            trace_root=trace_root,
            expected_uid=999,
        )

    assert result["available"] is True
    assert result["target_stable"] is True
    assert result["syscall"]["name"] == "read"
    assert result["syscall"]["fd"] == 7
    assert result["syscall"]["requested_bytes"] == 1837
    assert result["fd"]["target"].endswith("fixture.md")
    assert result["fd"]["stat"]["size"] == 7
    assert result["mountinfo"]["mount_id"] == 42
    assert "cifs" in result["mountinfo"]["matching_lines"][0]
    assert result["pending_mids_for_thread"] == [{"state": 2, "command": 8, "pid": 1235, "mid": 77}]
    assert "mid=77" in result["cifs_trace"]["pending_mid_matches"][0]
    assert result["smb_pcap_ring"]["available"] is True


def test_pcap_inventory_is_bounded_and_ignores_symlinks(tmp_path):
    (tmp_path / "client-smb.pcap0").write_bytes(b"pcap-0")
    (tmp_path / "client-smb.pcap1").write_bytes(b"pcap-1-longer")
    (tmp_path / "unrelated.pcap").write_bytes(b"ignore")
    (tmp_path / "client-smb.pcap-link").symlink_to(tmp_path / "client-smb.pcap0")

    with patch.object(
        forensics,
        "_run_command",
        return_value={"available": True, "raw": "ActiveState=active"},
    ):
        result = forensics._capture_pcap_ring(pcap_dir=tmp_path)

    assert result["available"] is True
    assert {Path(item["path"]).name for item in result["files"]} == {
        "client-smb.pcap0",
        "client-smb.pcap1",
    }
    assert result["total_bytes"] == len(b"pcap-0") + len(b"pcap-1-longer")


def test_collector_rejects_process_outside_worker_cgroup(tmp_path):
    proc_root = tmp_path / "proc"
    pid, tid = _make_fake_worker_proc(proc_root, cgroup="0::/user.slice/user-0.slice")

    with pytest.raises(forensics.ForensicsTargetError, match="kiara-worker.service"):
        forensics.collect_ingestion_stall_forensics(
            pid,
            tid,
            proc_root=proc_root,
            trace_root=tmp_path / "trace",
            expected_uid=999,
        )


def test_unprivileged_request_uses_fixed_sudo_command_and_parses_json():
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"available": True, "pending_mids": [{"mid": 77}]}),
            stderr="",
        )

    result = forensics.request_privileged_ingestion_forensics(1234, 1235, runner=runner)

    assert result["pending_mids"][0]["mid"] == 77
    assert calls[0][0] == [
        "/usr/bin/sudo",
        "-n",
        "/opt/kiara/venv/bin/kiara",
        "collect-ingestion-stall-forensics",
        "1234",
        "1235",
    ]
    assert calls[0][1]["timeout"] == 20
