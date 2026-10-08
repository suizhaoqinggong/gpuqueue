from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from gpuq.cli import request
from gpuq.store import JobStore


def test_store_allocates_multiple_gpus_atomically(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    first_id, first_token = store.create_job(
        uid=os.getuid(),
        username="tester",
        name="two-gpu",
        gpu_count=2,
        command=["python", "train.py"],
        cwd=str(tmp_path),
        log_path=None,
    )
    second_id, second_token = store.create_job(
        uid=os.getuid(),
        username="tester",
        name="one-gpu",
        gpu_count=1,
        command=["python", "train.py"],
        cwd=str(tmp_path),
        log_path=None,
    )

    store.register_supervisor(first_id, os.getuid(), first_token, 100)
    store.register_supervisor(second_id, os.getuid(), second_token, 101)
    store.schedule([0, 1, 2, 3], externally_busy=[3])

    assert store.get_job(first_id, os.getuid())["gpu_ids"] == [0, 1]
    assert store.get_job(second_id, os.getuid())["gpu_ids"] == [2]


def test_store_uses_strict_fifo(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    large_id, large_token = store.create_job(
        uid=os.getuid(),
        username="tester",
        name="large",
        gpu_count=2,
        command=["true"],
        cwd=str(tmp_path),
        log_path=None,
    )
    small_id, small_token = store.create_job(
        uid=os.getuid(),
        username="tester",
        name="small",
        gpu_count=1,
        command=["true"],
        cwd=str(tmp_path),
        log_path=None,
    )

    store.register_supervisor(large_id, os.getuid(), large_token, 100)
    store.register_supervisor(small_id, os.getuid(), small_token, 101)
    store.schedule([0, 1], externally_busy=[1])

    assert store.get_job(large_id, os.getuid())["state"] == "pending"
    assert store.get_job(small_id, os.getuid())["state"] == "pending"


def test_daemon_and_foreground_client(tmp_path: Path) -> None:
    # Unix-domain socket paths are limited to roughly 100 bytes on macOS/Linux.
    socket_dir = tempfile.TemporaryDirectory(prefix="gpuq-test-", dir="/tmp")
    socket_path = Path(socket_dir.name) / "gpuq.sock"
    state_dir = tmp_path / "state"
    env = os.environ.copy()
    env["GPUQ_GPU_IDS"] = "0,1"
    env["GPUQ_DAEMON_UID"] = str(os.getuid())
    env["GPUQ_TEST_HOST_KEY"] = str(tmp_path)
    env["GPUQ_CLIENT_STATE_DIR"] = str(tmp_path / "client")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    daemon = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpuq.daemon",
            "--socket",
            str(socket_path),
            "--state-dir",
            str(state_dir),
            "--interval",
            "0.05",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert socket_path.exists()

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpuq.cli",
                "--socket",
                str(socket_path),
                "run",
                "--",
                sys.executable,
                "-c",
                "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "allocated physical GPU(s) 0" in completed.stdout
        assert "\n0\n" in completed.stdout
        assert "succeeded" in completed.stdout

        detached = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpuq.cli",
                "--socket",
                str(socket_path),
                "submit",
                "--name",
                "detached-test",
                "--",
                sys.executable,
                "-c",
                "import os; print('detached=' + os.environ['CUDA_VISIBLE_DEVICES'])",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert detached.returncode == 0, detached.stdout + detached.stderr
        job_id = int(detached.stdout.split("submitted job ", 1)[1].split(";", 1)[0])

        deadline = time.monotonic() + 10
        # Client calls in this process use the same explicitly trusted daemon UID.
        from unittest.mock import patch

        with patch.dict(os.environ, {"GPUQ_DAEMON_UID": str(os.getuid())}):
            job = request({"action": "get", "job_id": job_id}, socket_path)["job"]
        while job["state"] not in {"succeeded", "failed", "cancelled", "lost"} and time.monotonic() < deadline:
            time.sleep(0.05)
            with patch.dict(os.environ, {"GPUQ_DAEMON_UID": str(os.getuid())}):
                job = request({"action": "get", "job_id": job_id}, socket_path)["job"]
        assert job["state"] == "succeeded"
        assert "detached=0" in Path(job["log_path"]).read_text()
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)
        socket_path.unlink(missing_ok=True)
        socket_dir.cleanup()


def test_command_logs_validation_and_tail(tmp_path: Path, monkeypatch: Any, capfd: Any) -> None:
    import argparse
    from gpuq.cli import ClientError, command_logs
    import gpuq.cli as cli

    # Negative lines rejected
    with pytest.raises(ClientError, match="nonnegative"):
        command_logs(argparse.Namespace(lines=-1, job_id=1, socket=tmp_path / "sock", follow=False))

    # Log does not exist rejected
    monkeypatch.setattr(cli, "request", lambda payload, sock: {"job": {"log_path": str(tmp_path / "absent.log")}})
    with pytest.raises(ClientError, match="does not exist"):
        command_logs(argparse.Namespace(lines=10, job_id=1, socket=tmp_path / "sock", follow=False))

    # Real file tailing
    log_file = tmp_path / "test.log"
    log_file.write_text("line1\nline2\nline3\nline4\n")
    monkeypatch.setattr(cli, "request", lambda payload, sock: {"job": {"log_path": str(log_file)}})

    # 0 lines returns 0 without error
    assert command_logs(argparse.Namespace(lines=0, job_id=1, socket=tmp_path / "sock", follow=False)) == 0

    # Normal tail
    command_logs(argparse.Namespace(lines=2, job_id=1, socket=tmp_path / "sock", follow=False))
    out, _ = capfd.readouterr()
    assert "line3" in out
    assert "line4" in out
    assert "line1" not in out
