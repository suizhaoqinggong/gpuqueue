"""Global allocation daemon for a single multi-GPU Linux host."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import signal
import socket
import socketserver
import stat
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gpu import GPU, discover_gpus, externally_busy_gpu_ids
from .security import peer_credentials, validate_socket_directory
from .store import TERMINAL_STATES, JobNotFoundError, JobPermissionError, JobStore

DEFAULT_SOCKET = Path(os.environ.get("GPUQ_SOCKET", "/run/gpuq/gpuq.sock"))
DEFAULT_STATE_DIR = Path(
    os.environ.get("GPUQ_STATE_DIR", str(Path.home() / ".local" / "state" / "gpuqd"))
)


class GPUQService:
    def __init__(self, store: JobStore, gpus: list[GPU], admin_uid: int, stale_after: float = 30.0) -> None:
        self.store = store
        self.gpus = gpus
        self.admin_uid = admin_uid
        self.stale_after = stale_after
        self.heartbeat_interval = max(0.5, min(2.0, stale_after / 4.0))

    def handle(self, uid: int, request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        action = request.get("action")
        if action == "ping":
            return {"ok": True, "gpus": len(self.gpus)}
        if action == "submit":
            command = request.get("command")
            if not isinstance(command, list) or not command or any(
                not isinstance(part, str) or "\x00" in part for part in command
            ):
                raise ValueError("command must be a nonempty list of strings without NUL bytes")
            gpu_count = int(request.get("gpu_count", 1))
            if gpu_count < 1 or gpu_count > len(self.gpus):
                raise ValueError(f"gpu_count must be between 1 and {len(self.gpus)}")
            username = pwd.getpwuid(uid).pw_name
            job_id, token = self.store.create_job(
                uid=uid,
                username=username,
                name=str(request.get("name") or "job"),
                gpu_count=gpu_count,
                command=command,
                cwd=str(request.get("cwd") or "."),
                log_path=str(request["log_path"]) if request.get("log_path") else None,
            )
            return {"ok": True, "job_id": job_id, "token": token}
        if action == "register":
            job = self.store.register_supervisor(
                int(request["job_id"]), uid, str(request["token"]), int(request["supervisor_pid"])
            )
            return self.job_response(job)
        if action == "heartbeat":
            job = self.store.heartbeat(int(request["job_id"]), uid, str(request["token"]))
            return self.job_response(job)
        if action == "started":
            job = self.store.mark_started(
                int(request["job_id"]), uid, str(request["token"]), int(request["process_pid"])
            )
            return self.job_response(job)
        if action == "finished":
            job = self.store.finish(
                int(request["job_id"]),
                uid,
                str(request["token"]),
                int(request["return_code"]),
                str(request.get("message") or ""),
            )
            return self.job_response(job)
        if action == "list":
            jobs = self.store.list_jobs(
                uid, include_all=bool(request.get("all", False)), admin_uid=self.admin_uid
            )
            decorated = [self.decorate_job(job) for job in jobs]
            if uid != self.admin_uid:
                sanitized = []
                for job in decorated:
                    if job["uid"] == uid:
                        sanitized.append(job)
                    else:
                        sanitized.append({
                            "id": job["id"],
                            "uid": job["uid"],
                            "username": job["username"],
                            "name": "-",
                            "state": job["state"],
                            "gpu_count": job["gpu_count"],
                            "gpu_ids": job.get("gpu_ids", []),
                            "gpu_uuids": job.get("gpu_uuids", []),
                            "created_at": job["created_at"],
                            "started_at": job.get("started_at"),
                            "cancel_requested": job.get("cancel_requested", False),
                        })
                return {"ok": True, "jobs": sanitized}
            return {"ok": True, "jobs": decorated}
        if action == "get":
            job = self.store.get_job(int(request["job_id"]), uid, admin_uid=self.admin_uid)
            return self.job_response(job)
        if action == "cancel":
            job = self.store.cancel(
                int(request["job_id"]),
                uid,
                admin_uid=self.admin_uid,
                force=bool(request.get("force", False)),
                reason=str(request.get("reason", "")),
            )
            return self.job_response(job)
        if action == "status":
            external = externally_busy_gpu_ids(self.gpus)
            allocations, allocation_states = self.store.active_allocations_summary()
            gpus = [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "memory_total_mib": gpu.memory_total_mib,
                    "external_busy": gpu.index in external and gpu.index not in allocations,
                    "job_id": allocations.get(gpu.index),
                    "job_state": allocation_states.get(allocations.get(gpu.index, -1)),
                }
                for gpu in self.gpus
            ]
            return {"ok": True, "gpus": gpus}
        raise ValueError(f"unknown action: {action!r}")

    def decorate_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        by_index = {gpu.index: gpu.uuid for gpu in self.gpus}
        result = dict(job)
        # Terminal records predate possible topology changes. Their numeric IDs
        # cannot safely identify today's cards until UUIDs are stored per job.
        result["gpu_uuids"] = (
            [] if job.get("state") in TERMINAL_STATES else [by_index[index] for index in job.get("gpu_ids", [])]
        )
        return result

    def job_response(self, job: Dict[str, Any]) -> Dict[str, Any]:
        response = {"ok": True, "job": self.decorate_job(job)}
        response["heartbeat_interval"] = self.heartbeat_interval
        return response


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            self.request.settimeout(5)
            _, uid, _ = peer_credentials(self.request)
            raw = self.rfile.readline(1024 * 1024 + 1)
            if not raw:
                return
            if len(raw) > 1024 * 1024 or not raw.endswith(b"\n"):
                raise ValueError("request exceeds size limit or is missing newline")
            request = json.loads(raw.decode("utf-8"))
            response = self.server.service.handle(uid, request)  # type: ignore[attr-defined]
        except (JobNotFoundError, JobPermissionError, KeyError, TypeError, ValueError) as error:
            response = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        except Exception as error:  # pragma: no cover - last-resort daemon boundary
            response = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        try:
            self.wfile.write((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))
        except OSError:
            pass


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: str, handler: type[RequestHandler], service: GPUQService) -> None:
        self.service = service
        self.request_slots = threading.BoundedSemaphore(64)
        super().__init__(path, handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self.request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


def scheduler_loop(
    service: GPUQService, stop_event: threading.Event, interval: float, stale_after: float, policy: str = "fifo"
) -> None:
    gpu_ids = [gpu.index for gpu in service.gpus]
    while not stop_event.is_set():
        try:
            external = externally_busy_gpu_ids(service.gpus)
            service.store.expire_stale(stale_after)
            service.store.schedule(gpu_ids, external, policy=policy)
        except Exception as error:
            print(f"gpuqd scheduler error: {error}", flush=True)
        stop_event.wait(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stale-after", type=float, default=30.0)
    parser.add_argument("--policy", choices=("fifo", "fit"), default="fifo")
    parser.add_argument(
        "--recover-job",
        type=int,
        default=None,
        help="offline maintenance: force-release a quarantined job (daemon must be stopped)",
    )
    parser.add_argument("--reason", default="", help="reason for offline job recovery")
    return parser.parse_args()


def _acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("another gpuqd instance is already using this state directory") from error
    return handle


def _remove_stale_socket(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError(f"refusing to replace non-socket or foreign socket: {path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        probe.connect(str(path))
    except ConnectionRefusedError:
        path.unlink()
    except (socket.timeout, TimeoutError) as error:
        raise RuntimeError(f"cannot confirm existing service state on {path}: probe timed out; socket retained") from error
    else:
        raise RuntimeError(f"a daemon is already listening on {path}")
    finally:
        probe.close()


def acquire_gpu_locks(gpus: List[GPU]) -> List[socket.socket]:
    """Hold one kernel lock per GPU, independent of daemon state/socket paths.

    Linux abstract sockets are scoped to a network namespace. Production daemons
    must run in the host namespace. Other platforms support fake-GPU tests only.
    """
    locks: List[socket.socket] = []
    fake = os.environ.get("GPUQ_GPU_IDS") is not None
    if not sys.platform.startswith("linux") and not fake:
        raise RuntimeError("production GPU locking requires Linux in the host network namespace")
    try:
        for gpu in sorted(gpus, key=lambda item: item.uuid):
            key = gpu.uuid + (os.environ.get("GPUQ_TEST_HOST_KEY", "") if fake else "")
            digest = hashlib.sha256(key.encode()).hexdigest()
            lock = socket.socket(socket.AF_UNIX if sys.platform.startswith("linux") else socket.AF_INET)
            locks.append(lock)
            if sys.platform.startswith("linux"):
                lock.bind("\x00gpuq-gpu-" + digest)
            else:
                lock.bind(("127.0.0.1", 20000 + int(digest[:8], 16) % 40000))
        return locks
    except OSError as error:
        for lock in locks:
            lock.close()
        raise RuntimeError("another gpuqd instance already manages an overlapping GPU") from error


def validate_gpu_topology(state_dir: Path, gpus: List[GPU], store: JobStore) -> None:
    """Reject index remapping while persisted leases still reserve physical cards."""
    path = state_dir / "gpu-topology.json"
    topology = {str(gpu.index): gpu.uuid for gpu in gpus}
    try:
        previous = json.loads(path.read_text())
    except FileNotFoundError:
        previous = None
    if previous == topology:
        return
    if store.active_allocations():
        raise RuntimeError(
            "GPU topology is missing or changed while active leases exist; "
            "restore the original GPU mapping and verify existing tasks before migrating state"
        )
    descriptor, temporary = tempfile.mkstemp(prefix=".gpu-topology-", dir=str(state_dir))
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(topology, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(state_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recover_offline(state_dir: Path, job_id: int, reason: str) -> None:
    if not reason.strip():
        raise ValueError("--reason is required for offline recovery")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_info = state_dir.lstat()
    if not stat.S_ISDIR(state_info.st_mode) or state_info.st_uid != os.getuid():
        raise PermissionError("state directory must be a real directory owned by the daemon")
    state_dir.chmod(0o700)
    lock_path = state_dir / "gpuqd.lock"
    lock_handle = _acquire_lock(lock_path)
    try:
        db_path = state_dir / "jobs.sqlite3"
        if not db_path.exists():
            raise FileNotFoundError(f"database file does not exist: {db_path}")
        store = JobStore(db_path)
        job = store.cancel(job_id, os.getuid(), admin_uid=os.getuid(), force=True, reason=reason)
        print(f"offline recovery complete: job {job_id} force-cancelled; state={job['state']}", flush=True)
    finally:
        lock_handle.close()


def main() -> int:
    args = parse_args()
    if args.recover_job is not None:
        recover_offline(args.state_dir, args.recover_job, args.reason)
        return 0
    if args.interval <= 0 or args.stale_after <= 0:
        raise ValueError("interval and stale-after must be positive")
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_info = args.state_dir.lstat()
    if not stat.S_ISDIR(state_info.st_mode) or state_info.st_uid != os.getuid():
        raise PermissionError("state directory must be a real directory owned by the daemon")
    args.state_dir.chmod(0o700)
    for state_file in ("gpuqd.lock", "jobs.sqlite3", "jobs.sqlite3-wal", "jobs.sqlite3-shm", "gpu-topology.json"):
        path = args.state_dir / state_file
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise PermissionError(f"state files must be owned regular files without links: {path}")
        path.chmod(0o600)
    lock_handle = _acquire_lock(args.state_dir / "gpuqd.lock")

    gpus = discover_gpus()
    if not gpus:
        raise RuntimeError("no GPUs detected")
    gpu_locks = acquire_gpu_locks(gpus)

    store = JobStore(args.state_dir / "jobs.sqlite3")
    validate_gpu_topology(args.state_dir, gpus, store)
    store.recover_active()
    service = GPUQService(store, gpus, os.getuid(), stale_after=args.stale_after)

    args.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    validate_socket_directory(args.socket, os.getuid())
    _remove_stale_socket(args.socket)
    server = ThreadingUnixServer(str(args.socket), RequestHandler, service)
    os.chmod(args.socket, 0o666)

    stop_event = threading.Event()
    scheduler = threading.Thread(
        target=scheduler_loop,
        args=(service, stop_event, args.interval, args.stale_after, args.policy),
        name="gpuq-scheduler",
        daemon=True,
    )
    scheduler.start()

    def stop(_signum: int, _frame: Optional[Any]) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"gpuqd listening on {args.socket}; GPUs=" + ",".join(str(gpu.index) for gpu in gpus),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        # Keep the lock file descriptor alive until the daemon has fully stopped.
        _ = lock_handle
        _ = gpu_locks
        stop_event.set()
        scheduler.join(timeout=5)
        server.server_close()
        try:
            args.socket.unlink()
        except FileNotFoundError:
            pass
        for gpu_lock in gpu_locks:
            gpu_lock.close()
        lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
