"""NVIDIA GPU discovery used by the queue daemon."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mib: int


def _run_nvidia_smi(*query_fields: str) -> str:
    query = ",".join(query_fields)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + query,
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def discover_gpus() -> List[GPU]:
    """Return the physical NVIDIA GPUs visible to the daemon.

    ``GPUQ_GPU_IDS`` is intentionally supported for tests and CPU-only
    development machines. It must not be set on the production daemon.
    """

    fake_ids = os.environ.get("GPUQ_GPU_IDS")
    if fake_ids is not None:
        indices = [int(value.strip()) for value in fake_ids.split(",") if value.strip()]
        return [GPU(index=index, uuid=f"GPUQ-FAKE-{index}", name="fake", memory_total_mib=0) for index in indices]

    output = _run_nvidia_smi("index", "uuid", "name", "memory.total")
    gpus: List[GPU] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 4:
            continue
        gpus.append(
            GPU(
                index=int(row[0].strip()),
                uuid=row[1].strip(),
                name=row[2].strip(),
                memory_total_mib=int(row[3].strip()),
            )
        )
    return sorted(gpus, key=lambda gpu: gpu.index)


def externally_busy_gpu_ids(gpus: List[GPU]) -> Set[int]:
    """Return unleased GPUs that currently have a CUDA compute process."""

    if os.environ.get("GPUQ_GPU_IDS") is not None:
        return set()

    uuid_to_index: Dict[str, int] = {gpu.uuid: gpu.index for gpu in gpus}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # Failing closed avoids double-allocating a card while nvidia-smi is
        # temporarily unavailable.
        return {gpu.index for gpu in gpus}

    busy: Set[int] = set()
    for row in csv.reader(io.StringIO(completed.stdout)):
        if not row:
            continue
        index = uuid_to_index.get(row[0].strip())
        if index is not None:
            busy.add(index)
    return busy
