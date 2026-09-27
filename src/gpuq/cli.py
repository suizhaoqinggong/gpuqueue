"""Command-line client for gpuq."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from .lifecycle import ProcessScope
from .security import verify_daemon

DEFAULT_SOCKET = Path(os.environ.get("GPUQ_SOCKET", "/run/gpuq/gpuq.sock"))
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "lost"}
STATE_LABELS = {
    "pending": "PD",
    "allocated": "AL",
    "running": "R",
    "succeeded": "CD",
    "failed": "F",
    "cancelled": "CA",
    "lost": "LO",
    "quarantined": "Q",
}


class ClientError(RuntimeError):
    pass


def request(payload: Dict[str, Any], socket_path: Path = DEFAULT_SOCKET) -> Dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(10)
        client.connect(str(socket_path))
        verify_daemon(client, socket_path)
        client.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        chunks: List[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > 4 * 1024 * 1024:
                raise ClientError("gpuqd response exceeds size limit")
            if b"\n" in chunk:
                break
    except OSError as error:
        raise ClientError(f"cannot reach gpuqd at {socket_path}: {error}") from error
    finally:
        client.close()
    if not chunks:
        raise ClientError("gpuqd returned an empty response")
    decoded = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ClientError("gpuqd returned an invalid response")
    response = cast(Dict[str, Any], decoded)
    if not response.get("ok"):
        raise ClientError(str(response.get("error", "gpuqd request failed")))
    return response


def _strip_separator(command: Sequence[str]) -> List[str]:
    result = list(command)
    if result and result[0] == "--":
        result.pop(0)
    if not result:
        raise ClientError("a command is required after --")
    return result


def _log_path() -> Path:
    root = Path(os.environ.get("GPUQ_CLIENT_STATE_DIR", str(Path.home() / ".local" / "state" / "gpuq")))
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = log_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.log"
    path.touch(mode=0o600, exist_ok=False)
    return path


def _create_job(args: argparse.Namespace, command: Sequence[str], *, log_path: Optional[Path]) -> tuple[int, str]:
    response = request(
        {
            "action": "submit",
            "name": args.name or Path(command[0]).name,
            "gpu_count": args.gpus,
            "command": list(command),
            "cwd": str(Path.cwd()),
            "log_path": str(log_path) if log_path else None,
        },
        args.socket,
    )
    return int(response["job_id"]), str(response["token"])


def _job_request(action: str, job_id: int, token: str, socket_path: Path, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"action": action, "job_id": job_id, "token": token}
    payload.update(extra)
    job = request(payload, socket_path)["job"]
    if not isinstance(job, dict):
        raise ClientError("gpuqd returned an invalid job")
    return cast(Dict[str, Any], job)


def _mirror_log(path: Path, stop: threading.Event) -> None:
    # Console backpressure must never block the lease heartbeat or cancellation.
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(65536)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                elif stop.is_set():
                    return
                else:
                    stop.wait(0.1)
    except (OSError, ValueError):
        return


def supervise(
    *,
    job_id: int,
    token: str,
    socket_path: Path,
    interactive: bool,
    mirror_output: bool,
    local_job: Dict[str, Any],
) -> int:
    scope = ProcessScope()
    process: Optional[subprocess.Popen[bytes]] = None
    log_handle: Optional[Any] = None
    old_foreground_pgid: Optional[int] = None
    old_sigttou: Any = None
    gate_read: Optional[int] = None
    gate_write: Optional[int] = None
    mirror_stop = threading.Event()
    mirror_thread: Optional[threading.Thread] = None
    stopped = threading.Event()
    registered = False
    old_signals = {}

    def stop_handler(signum: int, frame: Any) -> None:
        stopped.set()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_signals[signum] = signal.signal(signum, stop_handler)

    try:
        job = _job_request("register", job_id, token, socket_path, supervisor_pid=os.getpid())
        registered = True
        while job["state"] == "pending":
            if stopped.wait(0.5):
                request({"action": "cancel", "job_id": job_id}, socket_path)
                _job_request("finished", job_id, token, socket_path, return_code=130)
                return 130
            job = _job_request("heartbeat", job_id, token, socket_path)
        if job["state"] in TERMINAL_STATES:
            return int(job.get("return_code") or (0 if job["state"] == "succeeded" else 1))
        if job["state"] != "allocated" or job["cancel_requested"] or stopped.is_set():
            raise ClientError(f"job {job_id} entered unexpected state {job['state']}")

        gpu_ids = [int(value) for value in job["gpu_ids"]]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        devices = job.get("gpu_uuids")
        if not devices or any(str(value).startswith("GPUQ-FAKE-") for value in devices):
            if "GPUQ_GPU_IDS" not in env:
                raise ClientError("daemon must supply GPU UUIDs")
            devices = [str(value) for value in gpu_ids]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        env["GPUQ_JOB_ID"] = str(job_id)
        print(
            f"job {job_id}: allocated physical GPU(s) {','.join(map(str, gpu_ids))} "
            f"as logical cuda:0..{len(gpu_ids) - 1}",
            flush=True,
        )

        # Execution inputs stay with the submitting user, never from an RPC reply.
        command = [str(part) for part in local_job["command"]]
        gate_read, gate_write = os.pipe()
        gated_command = [sys.executable, "-m", "gpuq.startgate", str(gate_read)] + command
        if interactive:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise ClientError("gpuq shell requires an interactive terminal")
            stdin_fd = sys.stdin.fileno()
            old_foreground_pgid = os.tcgetpgrp(stdin_fd)
            old_sigttou = signal.getsignal(signal.SIGTTOU)
            signal.signal(signal.SIGTTOU, signal.SIG_IGN)

            def prepare_interactive_child() -> None:
                os.setpgrp()
                os.tcsetpgrp(stdin_fd, os.getpgrp())

            process = subprocess.Popen(
                gated_command,
                cwd=local_job["cwd"],
                env=env,
                preexec_fn=prepare_interactive_child,
                pass_fds=(gate_read,),
            )
        else:
            log_path = Path(local_job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab", buffering=0)
            process = subprocess.Popen(
                gated_command,
                cwd=local_job["cwd"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(gate_read,),
            )

        os.close(gate_read)
        gate_read = None
        job = _job_request("started", job_id, token, socket_path, process_pid=process.pid)
        if job["state"] != "running" or job["cancel_requested"] or stopped.is_set():
            raise ClientError("lease revoked before command startup")
        os.write(gate_write, b"1")
        os.close(gate_write)
        gate_write = None
        if mirror_output:
            mirror_thread = threading.Thread(target=_mirror_log, args=(log_path, mirror_stop), daemon=True)
            mirror_thread.start()
        cancel_sent = False
        warned = False
        idle_since = time.monotonic()
        idle_limit = float(local_job.get("idle_timeout", 0))
        while not scope.empty(process):
            if interactive and idle_limit:
                if scope.shell_has_children(process):
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= idle_limit:
                    print(f"gpuq: shell has had no child commands for {idle_limit:g}s; closing", flush=True)
                    stopped.set()
            if stopped.is_set() and not cancel_sent:
                try:
                    request({"action": "cancel", "job_id": job_id}, socket_path)
                    cancel_sent = True
                except ClientError:
                    pass
            try:
                job = _job_request("heartbeat", job_id, token, socket_path)
            except ClientError as error:
                if not warned:
                    print(f"gpuq warning: {error}; lease remains reserved", file=sys.stderr, flush=True)
                    warned = True
            if (stopped.is_set() or job["cancel_requested"] or job["state"] != "running"
                    or process.poll() is not None):
                scope.stop_step(process)
            time.sleep(0.2)

        return_code = int(process.wait())
        # Retry completion after daemon downtime; never silently discard it.
        while True:
            try:
                final = _job_request("finished", job_id, token, socket_path, return_code=return_code)
                break
            except ClientError as error:
                print(f"gpuq: completion pending: {error}", file=sys.stderr, flush=True)
                time.sleep(2)
        mirror_stop.set()
        if mirror_thread is not None:
            mirror_thread.join(timeout=1)
        print(f"job {job_id}: {final['state']} (exit={return_code})", flush=True)
        return return_code if return_code >= 0 else 128 - return_code
    except BaseException as error:
        if gate_write is not None:
            os.close(gate_write)
            gate_write = None
        if process is not None:
            while not scope.empty(process):
                scope.stop_step(process)
                try:
                    _job_request("heartbeat", job_id, token, socket_path)
                except Exception:
                    pass
                time.sleep(0.2)
        if registered:
            while True:
                try:
                    _job_request("finished", job_id, token, socket_path, return_code=125, message=str(error))
                    break
                except ClientError as finish_error:
                    print(f"gpuq: cleanup complete; awaiting daemon: {finish_error}", file=sys.stderr, flush=True)
                    time.sleep(2)
        raise
    finally:
        for fd in (gate_read, gate_write):
            if fd is not None:
                os.close(fd)
        mirror_stop.set()
        if old_foreground_pgid is not None:
            try:
                os.tcsetpgrp(sys.stdin.fileno(), old_foreground_pgid)
            except OSError:
                pass
        if old_sigttou is not None:
            signal.signal(signal.SIGTTOU, old_sigttou)
        if log_handle is not None:
            log_handle.close()
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)


def command_run(args: argparse.Namespace) -> int:
    command = _strip_separator(args.command)
    log_path = _log_path()
    job_id, token = _create_job(args, command, log_path=log_path)
    print(f"submitted job {job_id}; log={log_path}")
    return supervise(
        job_id=job_id,
        token=token,
        socket_path=args.socket,
        interactive=False,
        mirror_output=True,
        local_job={"command": command, "cwd": str(Path.cwd()), "log_path": str(log_path)},
    )


def command_submit(args: argparse.Namespace) -> int:
    command = _strip_separator(args.command)
    log_path = _log_path()
    job_id, token = _create_job(args, command, log_path=log_path)
    child_command = [
        sys.executable,
        "-m",
        "gpuq.cli",
        "_supervise",
        "--socket",
        str(args.socket),
    ]
    try:
        with log_path.open("ab", buffering=0) as diagnostics:
            child = subprocess.Popen(
                child_command, stdin=subprocess.PIPE, stdout=diagnostics,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            )
            assert child.stdin is not None
            child.stdin.write(json.dumps({"job_id": job_id, "token": token, "command": command,
                                         "cwd": str(Path.cwd()), "log_path": str(log_path)}).encode())
            child.stdin.close()
    except OSError as error:
        request({"action": "cancel", "job_id": job_id}, args.socket)
        raise ClientError(f"could not start the user-side supervisor: {error}") from error
    print(f"submitted job {job_id}; log={log_path}")
    return 0


def command_shell(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ClientError("gpuq shell requires an interactive terminal")
    if args.idle_timeout < 0:
        raise ClientError("idle-timeout must be nonnegative")
    shell = args.shell or os.environ.get("SHELL") or "/bin/bash"
    command = [shell, "-l"]
    job_id, token = _create_job(args, command, log_path=None)
    return supervise(
        job_id=job_id,
        token=token,
        socket_path=args.socket,
        interactive=True,
        mirror_output=False,
        local_job={"command": command, "cwd": str(Path.cwd()), "log_path": None,
                   "idle_timeout": args.idle_timeout},
    )


def _format_age(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "-"
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def command_list(args: argparse.Namespace) -> int:
    jobs = request({"action": "list", "all": args.all}, args.socket)["jobs"]
    print(f"{'JOB':>6} {'ST':>2} {'USER':<12} {'GPU':<8} {'AGE':>5} NAME")
    for job in jobs:
        gpu_text = ",".join(str(value) for value in job["gpu_ids"]) or f"{job['gpu_count']} req"
        print(
            f"{job['id']:>6} {STATE_LABELS.get(job['state'], '?'):>2} "
            f"{job['username']:<12.12} {gpu_text:<8.8} {_format_age(job['created_at']):>5} {job['name']}"
        )
    return 0


def command_status(args: argparse.Namespace) -> int:
    gpus = request({"action": "status"}, args.socket)["gpus"]
    print(f"{'GPU':>3} {'STATE':<10} {'JOB':>6} {'MEM':>9} NAME")
    for gpu in gpus:
        if gpu["job_id"] is not None:
            state = "quarantine" if gpu.get("job_state") == "quarantined" else "allocated"
        elif gpu["external_busy"]:
            state = "external"
        else:
            state = "idle"
        job = str(gpu["job_id"]) if gpu["job_id"] is not None else "-"
        print(f"{gpu['index']:>3} {state:<10} {job:>6} {gpu['memory_total_mib']:>6} MiB {gpu['name']}")
    return 0


def command_show(args: argparse.Namespace) -> int:
    job = request({"action": "get", "job_id": args.job_id}, args.socket)["job"]
    print(json.dumps(job, indent=2, ensure_ascii=False))
    return 0


def command_cancel(args: argparse.Namespace) -> int:
    job = request({"action": "cancel", "job_id": args.job_id}, args.socket)["job"]
    print(f"job {job['id']}: cancel requested; state={job['state']}")
    return 0


def command_logs(args: argparse.Namespace) -> int:
    job = request({"action": "get", "job_id": args.job_id}, args.socket)["job"]
    if not job.get("log_path"):
        raise ClientError(f"job {args.job_id} has no captured log")
    path = Path(job["log_path"])
    if args.follow:
        return subprocess.call(["tail", "-n", str(args.lines), "-f", str(path)])
    if not path.exists():
        raise ClientError(f"log does not exist yet: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-args.lines :]))
    return 0


def command_supervise(args: argparse.Namespace) -> int:
    config = json.load(sys.stdin)
    return supervise(
        job_id=int(config["job_id"]),
        token=str(config["token"]),
        socket_path=args.socket,
        interactive=False,
        mirror_output=False,
        local_job=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gpuq", description=__doc__)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    def add_job_options(subparser: argparse.ArgumentParser, *, command: bool) -> None:
        subparser.add_argument("-g", "--gpus", type=int, default=1)
        subparser.add_argument("-n", "--name", default=None)
        if command:
            subparser.add_argument("command", nargs=argparse.REMAINDER)

    run_parser = subparsers.add_parser("run", help="queue and run a command in the foreground")
    add_job_options(run_parser, command=True)
    run_parser.set_defaults(func=command_run)

    submit_parser = subparsers.add_parser("submit", help="queue a detached command")
    add_job_options(submit_parser, command=True)
    submit_parser.set_defaults(func=command_submit)

    shell_parser = subparsers.add_parser("shell", help="open a shell after GPUs are allocated")
    add_job_options(shell_parser, command=False)
    shell_parser.add_argument("--shell", default=None)
    shell_parser.add_argument("--idle-timeout", type=float, default=1800,
                              help="close shell after this many seconds without child commands (0 disables)")
    shell_parser.set_defaults(func=command_shell)

    list_parser = subparsers.add_parser("ls", aliases=["list"], help="list jobs")
    list_parser.add_argument("--all", action="store_true")
    list_parser.set_defaults(func=command_list)

    status_parser = subparsers.add_parser("status", help="show physical GPU allocation")
    status_parser.set_defaults(func=command_status)

    show_parser = subparsers.add_parser("show", help="show one job")
    show_parser.add_argument("job_id", type=int)
    show_parser.set_defaults(func=command_show)

    cancel_parser = subparsers.add_parser("cancel", help="cancel a job")
    cancel_parser.add_argument("job_id", type=int)
    cancel_parser.set_defaults(func=command_cancel)

    logs_parser = subparsers.add_parser("logs", help="show a job log")
    logs_parser.add_argument("job_id", type=int)
    logs_parser.add_argument("-f", "--follow", action="store_true")
    logs_parser.add_argument("-n", "--lines", type=int, default=100)
    logs_parser.set_defaults(func=command_logs)

    supervisor_parser = subparsers.add_parser("_supervise", help=argparse.SUPPRESS)
    supervisor_parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    supervisor_parser.set_defaults(func=command_supervise)

    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        return int(args.func(args))
    except ClientError as error:
        print(f"gpuq: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
