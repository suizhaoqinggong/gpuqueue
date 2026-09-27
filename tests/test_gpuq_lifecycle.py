"""CPU-only integration coverage for Linux job ownership and lease recovery."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

from gpuq.cli import ClientError, request

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux subreaper and /proc")


def wait_for(check: Callable[[], Any], timeout: float = 12.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("condition did not become true before timeout")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class QueueHarness:
    def __init__(self, root: Path, env: Dict[str, str]) -> None:
        self.root = root
        self.env = env
        self.socket = root / "queue.sock"
        self.daemon: Any = None
        self.clients: List[subprocess.Popen[Any]] = []
        self.worker_pids: List[int] = []

    def rpc(self, action: str, **values: Any) -> Dict[str, Any]:
        return request({"action": action, **values}, self.socket)

    def start_daemon(self) -> None:
        with (self.root / "daemon.log").open("ab") as output:
            self.daemon = subprocess.Popen(
                [sys.executable, "-m", "gpuq.daemon", "--socket", str(self.socket),
                 "--state-dir", str(self.root / "state"), "--interval", "0.05", "--stale-after", "2"],
                env=self.env, stdout=output, stderr=subprocess.STDOUT,
            )

        def ready() -> bool:
            assert self.daemon.poll() is None, (self.root / "daemon.log").read_text()
            try:
                return bool(self.rpc("ping")["ok"])
            except ClientError:
                return False

        wait_for(ready)

    def stop_daemon(self) -> None:
        if self.daemon is not None and self.daemon.poll() is None:
            self.daemon.terminate()
            self.daemon.wait(timeout=8)

    def start_job(self, code: str) -> Tuple[subprocess.Popen[Any], int]:
        with (self.root / f"client-{len(self.clients)}.log").open("ab") as output:
            client = subprocess.Popen(
                [sys.executable, "-m", "gpuq.cli", "--socket", str(self.socket), "run", "--",
                 sys.executable, "-c", code],
                env=self.env, stdout=output, stderr=subprocess.STDOUT,
            )
        self.clients.append(client)

        def registered() -> Any:
            return next((job for job in self.rpc("list")["jobs"] if job["supervisor_pid"] == client.pid), None)

        job = wait_for(registered)
        return client, int(job["id"])

    def job(self, job_id: int) -> Dict[str, Any]:
        return self.rpc("get", job_id=job_id)["job"]

    def state(self, job_id: int, expected: str) -> Dict[str, Any]:
        return wait_for(lambda: self.job(job_id) if self.job(job_id)["state"] == expected else None)

    def worker(self, marker: Path) -> int:
        wait_for(lambda: marker.exists() and bool(marker.read_text().strip()))
        pid = int(marker.read_text())
        self.worker_pids.append(pid)
        return pid

    def close(self) -> None:
        # Resume deliberately stopped supervisors so their own cleanup can run.
        for client in self.clients:
            if client.poll() is None:
                os.kill(client.pid, signal.SIGCONT)
                client.terminate()
        for client in self.clients:
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait(timeout=3)
        for pid in self.worker_pids:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        self.stop_daemon()


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[QueueHarness]:
    monkeypatch.setenv("GPUQ_DAEMON_UID", str(os.getuid()))
    with tempfile.TemporaryDirectory(prefix="gq-life-", dir="/tmp") as directory:
        root = Path(directory)
        env = os.environ.copy()
        env.update({
            "GPUQ_GPU_IDS": "0",
            "GPUQ_TEST_HOST_KEY": uuid.uuid4().hex,
            "GPUQ_CLIENT_STATE_DIR": str(root / "client-state"),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        })
        harness = QueueHarness(root, env)
        try:
            harness.start_daemon()
            yield harness
        finally:
            harness.close()


def tree_command(marker: Path, *, leader_exits: bool = False) -> str:
    # The worker escapes its parent's process group and refuses graceful stop.
    # It also retains stdout, which used to block the supervisor's final drain.
    worker = (
        "import os,signal,time; from pathlib import Path; "
        "os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    return (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{worker!r}]); "
        f"marker=Path({str(marker)!r}); "
        "\nwhile not marker.exists(): time.sleep(0.01)\n"
        + ("" if leader_exits else "time.sleep(60)\n")
    )


def test_cancel_cleans_term_resistant_worker_in_separate_session(queue: QueueHarness) -> None:
    marker = queue.root / "cancel-worker.pid"
    client, job_id = queue.start_job(tree_command(marker))
    pid = queue.worker(marker)
    assert os.getsid(pid) == pid
    queue.rpc("cancel", job_id=job_id)
    queue.state(job_id, "cancelled")
    client.wait(timeout=5)
    assert not alive(pid)
    assert queue.rpc("status")["gpus"][0]["job_id"] is None


def test_leader_exit_cleans_all_descendants_before_releasing_lease(queue: QueueHarness) -> None:
    marker = queue.root / "orphan-worker.pid"
    client, job_id = queue.start_job(tree_command(marker, leader_exits=True))
    pid = queue.worker(marker)
    deadline = time.monotonic() + 10
    while client.poll() is None and time.monotonic() < deadline:
        if alive(pid):
            assert queue.rpc("status")["gpus"][0]["job_id"] == job_id
        else:
            break
        time.sleep(0.05)
    assert client.wait(timeout=10) == 0
    assert not alive(pid)
    assert queue.job(job_id)["state"] == "succeeded"


def test_stopped_supervisor_quarantines_card_until_resumed_cleanup(queue: QueueHarness) -> None:
    marker = queue.root / "paused-worker.pid"
    client, job_id = queue.start_job(tree_command(marker))
    pid = queue.worker(marker)
    os.kill(client.pid, signal.SIGSTOP)
    try:
        queue.state(job_id, "quarantined")
        assert alive(pid)
        assert queue.rpc("status")["gpus"][0]["job_id"] == job_id
        _, queued_id = queue.start_job("import time; time.sleep(1)")
        time.sleep(0.3)
        assert queue.job(queued_id)["state"] == "pending"
    finally:
        os.kill(client.pid, signal.SIGCONT)
    client.wait(timeout=10)
    assert not alive(pid)
    wait_for(lambda: queue.job(queued_id)["state"] != "pending")


def test_daemon_restart_preserves_running_lease(queue: QueueHarness) -> None:
    marker = queue.root / "restart-worker.pid"
    client, job_id = queue.start_job(tree_command(marker))
    pid = queue.worker(marker)
    queue.stop_daemon()
    queue.start_daemon()
    assert queue.job(job_id)["state"] == "running"
    assert queue.rpc("status")["gpus"][0]["job_id"] == job_id
    assert alive(pid)
    before = queue.job(job_id)["heartbeat_at"]
    wait_for(lambda: queue.job(job_id)["heartbeat_at"] > before)
    queue.rpc("cancel", job_id=job_id)
    queue.state(job_id, "cancelled")
    client.wait(timeout=5)
    assert not alive(pid)


def test_inherited_stdout_does_not_interrupt_cleanup_heartbeats(queue: QueueHarness) -> None:
    marker = queue.root / "stdout-worker.pid"
    client, job_id = queue.start_job(tree_command(marker, leader_exits=True))
    pid = queue.worker(marker)
    heartbeat = queue.job(job_id)["heartbeat_at"]
    # Killing the TERM-resistant worker takes 3 seconds, beyond the 2-second
    # stale threshold. Heartbeats must continue while its inherited FD is open.
    time.sleep(2.2)
    job = queue.job(job_id)
    assert job["state"] == "running"
    assert job["heartbeat_at"] > heartbeat + 1
    assert alive(pid)
    assert client.wait(timeout=10) == 0
    assert not alive(pid)
    assert queue.job(job_id)["state"] == "succeeded"
