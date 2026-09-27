"""Track a Linux job tree, including children which create new sessions."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class ProcessScope:
    """One supervisor owns one task. Never release a lease while children remain."""

    def __init__(self) -> None:
        self.linux = sys.platform.startswith("linux")
        if self.linux:
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
        elif "GPUQ_GPU_IDS" not in os.environ:
            raise RuntimeError("production supervisors require Linux")
        self.stopping_at: Optional[float] = None

    @staticmethod
    def _stat(pid: int) -> Optional[Tuple[int, str]]:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            return int(fields[1]), fields[19]  # ppid, starttime (PID reuse guard)
        except (FileNotFoundError, ProcessLookupError):
            return None

    def _descendants(self) -> Dict[int, str]:
        table = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                value = self._stat(int(path.name))
                if value is not None:
                    table[int(path.name)] = value
        parents = {os.getpid()}
        result: Dict[int, str] = {}
        while parents:
            children = {pid for pid, (ppid, _) in table.items() if ppid in parents and pid not in result}
            for pid in children:
                result[pid] = table[pid][1]
            parents = children
        return result

    def empty(self, process: subprocess.Popen[bytes]) -> bool:
        # Let Popen reap its direct child and retain the actual command exit code.
        if process.poll() is None:
            return False
        if not self.linux:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            return False
        # ECHILD is the authoritative emptiness check. A /proc snapshot alone
        # could miss a concurrent fork/reparent. Subreaping keeps orphans here.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return True
            if pid == 0:
                return False

    def shell_has_children(self, process: subprocess.Popen[bytes]) -> bool:
        return not self.linux or any(pid != process.pid for pid in self._descendants())

    def stop_step(self, process: subprocess.Popen[bytes], grace_seconds: float = 3.0) -> None:
        if self.stopping_at is None:
            self.stopping_at = time.monotonic()
        signum = signal.SIGTERM if time.monotonic() - self.stopping_at < grace_seconds else signal.SIGKILL
        if self.linux:
            for pid, started in self._descendants().items():
                current = self._stat(pid)
                if current is not None and current[1] == started:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
        else:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
