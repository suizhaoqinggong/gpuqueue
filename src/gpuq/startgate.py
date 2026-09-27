"""Exec a locally supplied command only after the supervisor confirms its lease."""

import os
import sys


def main() -> int:
    fd = int(sys.argv[1])
    try:
        allowed = os.read(fd, 1) == b"1"
    finally:
        os.close(fd)
    if not allowed:
        return 125
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
