# gpuq

Single-host cooperative queue for NVIDIA GPU allocation. The daemon grants
leases; each user's supervisor launches their own command. The runtime uses
only the Python standard library.

Linux and NVIDIA GPUs are required for a real deployment. Python 3.9 or newer
is supported. macOS can run the state, security, and client tests.

## Layout

```text
src/gpuq/     queue package
tests/        pytest suite
deploy/       client wrapper, daemon wrapper, systemd unit
```

## Install

```bash
pip install -e '.[dev]'
```

## Verify

```bash
python -m pytest -q
```

Lifecycle integration tests require Linux and use simulated GPUs.

## Run

See [docs/gpuq.md](docs/gpuq.md) for scheduling policy, failure behavior, and
the server-wide install.
