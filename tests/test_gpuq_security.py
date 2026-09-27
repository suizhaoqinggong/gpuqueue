from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from gpuq.daemon import GPUQService, _remove_stale_socket, acquire_gpu_locks, validate_gpu_topology
from gpuq.gpu import GPU
from gpuq.security import validate_socket_directory, verify_daemon
from gpuq.store import JobStore


def test_overlapping_daemons_cannot_lock_same_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPUQ_GPU_IDS", "0,1")
    monkeypatch.setenv("GPUQ_TEST_HOST_KEY", str(tmp_path))
    gpu = GPU(0, "GPUQ-FAKE-0", "fake", 0)
    locks = acquire_gpu_locks([gpu])
    try:
        with pytest.raises(RuntimeError, match="overlapping GPU"):
            acquire_gpu_locks([GPU(3, gpu.uuid, "fake", 0)])
    finally:
        for lock in locks:
            lock.close()
    recovered = acquire_gpu_locks([gpu])
    for lock in recovered:
        lock.close()


def test_shared_socket_parent_is_rejected(tmp_path: Path) -> None:
    tmp_path.chmod(0o777)
    with pytest.raises(PermissionError, match="not writable"):
        validate_socket_directory(tmp_path / "daemon.sock", os.getuid())


def test_stale_socket_cleanup_preserves_regular_files_and_symlinks(tmp_path: Path) -> None:
    regular = tmp_path / "file"
    regular.touch()
    link = tmp_path / "link"
    link.symlink_to(regular)
    for path in (regular, link):
        with pytest.raises(PermissionError, match="refusing to replace"):
            _remove_stale_socket(path)
        assert path.exists()


def test_peer_identity_must_match_trusted_uid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A socket pair provides real kernel credentials without path-length limits.
    local, peer = socket.socketpair()
    socket_path = tmp_path / "daemon.sock"
    # Validate the real peer separately from filesystem UID by mocking only stat.
    from types import SimpleNamespace

    import gpuq.security as security

    monkeypatch.setattr(security, "validate_socket_directory", lambda path, uid: None)
    monkeypatch.setattr(Path, "lstat", lambda self: SimpleNamespace(st_mode=0o140600, st_uid=os.getuid() + 1))
    try:
        with pytest.raises(PermissionError, match="peer UID"):
            verify_daemon(local, socket_path, expected_uid=os.getuid() + 1)
    finally:
        local.close()
        peer.close()


@pytest.mark.parametrize("command", ["python train.py", [], [1], ["python", "bad\x00arg"]])
def test_submit_rejects_malformed_command(tmp_path: Path, command: object) -> None:
    service = GPUQService(JobStore(tmp_path / "jobs.db"), [GPU(0, "GPU-actual", "fake", 0)], os.getuid())
    with pytest.raises(ValueError, match="command must"):
        service.handle(os.getuid(), {"action": "submit", "command": command})


def test_job_response_includes_stable_gpu_uuid(tmp_path: Path) -> None:
    service = GPUQService(JobStore(tmp_path / "jobs.db"), [GPU(2, "GPU-actual", "fake", 0)], os.getuid())
    assert service.job_response({"gpu_ids": [2]})["job"]["gpu_uuids"] == ["GPU-actual"]
    with pytest.raises(ValueError, match="JSON object"):
        service.handle(os.getuid(), [])  # type: ignore[arg-type]


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled", "lost"])
def test_terminal_job_does_not_use_current_topology(tmp_path: Path, state: str) -> None:
    service = GPUQService(JobStore(tmp_path / "jobs.db"), [GPU(2, "GPU-new", "fake", 0)], os.getuid())
    # Both removed indices and reused indices must avoid reporting a new card
    # as the physical GPU that an old, completed job used.
    for old_ids in ([0], [2]):
        result = service.job_response({"gpu_ids": old_ids, "state": state})["job"]
        assert result["gpu_ids"] == old_ids
        assert result["gpu_uuids"] == []


def test_topology_change_cannot_remap_active_lease(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    gpus = [GPU(0, "GPU-first", "fake", 0), GPU(1, "GPU-second", "fake", 0)]
    validate_gpu_topology(tmp_path, gpus, store)
    job_id, token = store.create_job(
        uid=os.getuid(), username="test", name="test", gpu_count=1,
        command=["true"], cwd=str(tmp_path), log_path=None,
    )
    store.register_supervisor(job_id, os.getuid(), token, os.getpid())
    store.schedule([0, 1], [])
    validate_gpu_topology(tmp_path, gpus, store)
    swapped = [GPU(1, "GPU-first", "fake", 0), GPU(0, "GPU-second", "fake", 0)]
    with pytest.raises(RuntimeError, match="topology"):
        validate_gpu_topology(tmp_path, swapped, store)
    store.finish(job_id, os.getuid(), token, 0)
    validate_gpu_topology(tmp_path, swapped, store)


def test_missing_topology_cannot_adopt_legacy_active_leases(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job_id, token = store.create_job(
        uid=os.getuid(), username="test", name="test", gpu_count=1,
        command=["true"], cwd=str(tmp_path), log_path=None,
    )
    store.register_supervisor(job_id, os.getuid(), token, os.getpid())
    store.schedule([0], [])
    with pytest.raises(RuntimeError, match="topology"):
        validate_gpu_topology(tmp_path, [GPU(0, "GPU-first", "fake", 0)], store)
