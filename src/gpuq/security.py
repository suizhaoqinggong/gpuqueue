"""Local peer authentication and filesystem trust boundaries."""

from __future__ import annotations

import os
import socket
import stat
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple


def peer_credentials(connection: socket.socket) -> Tuple[int, int, int]:
    """Return kernel-supplied peer credentials; never guess a peer identity."""
    if hasattr(socket, "SO_PEERCRED"):
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", raw)
    getpeereid = getattr(connection, "getpeereid", None)
    if getpeereid is not None:
        uid, gid = getpeereid()
        return 0, int(uid), int(gid)
    if sys.platform == "darwin":
        # CPython does not expose getpeereid on every macOS build.
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        native_getpeereid = libc.getpeereid
        native_getpeereid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        native_getpeereid.restype = ctypes.c_int
        peer_uid, peer_gid = ctypes.c_uint(), ctypes.c_uint()
        if native_getpeereid(connection.fileno(), ctypes.byref(peer_uid), ctypes.byref(peer_gid)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return 0, peer_uid.value, peer_gid.value
    raise RuntimeError("this platform cannot authenticate Unix socket peers")


def trusted_daemon_uid() -> int:
    return int(os.environ.get("GPUQ_DAEMON_UID", "0"))


def validate_socket_directory(path: Path, uid: int) -> None:
    directory = path.parent
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, uid) or info.st_mode & 0o022:
        raise PermissionError(
            f"socket directory must be owned by daemon/root (UID 0 or {uid}, got {info.st_uid}) "
            f"and not writable by others (mode {oct(info.st_mode)}): {directory}. "
            f"Suggest using /run/gpuq or a private user directory."
        )
    # Ancestors may include a sticky shared /tmp, but not an unprotected shared directory.
    for ancestor in directory.resolve().parents:
        info = ancestor.stat()
        if info.st_uid not in (0, uid) or (info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX):
            raise PermissionError(
                f"untrusted socket directory ancestor {ancestor}: owned by UID {info.st_uid} (expected 0 or {uid}), "
                f"mode {oct(info.st_mode)}. If using shared storage/mounts, ensure parent directories are owned by root/user "
                f"or use /run/gpuq."
            )


def verify_daemon(connection: socket.socket, socket_path: Path, expected_uid: Optional[int] = None) -> None:
    uid = trusted_daemon_uid() if expected_uid is None else expected_uid
    validate_socket_directory(socket_path, uid)
    info = socket_path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != uid:
        raise PermissionError(f"socket is not owned by trusted daemon UID {uid}")
    _, actual_uid, _ = peer_credentials(connection)
    if actual_uid != uid:
        raise PermissionError(f"daemon peer UID {actual_uid} does not match trusted UID {uid}")
