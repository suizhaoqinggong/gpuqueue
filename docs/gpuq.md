# gpuq: single-host GPU queue

Linux, NVIDIA GPUs, Python 3.9 recommended, no third-party runtime dependency.
The daemon allocates leases; each user's supervisor launches their own command.
All daemons must run in the host network namespace. This remains a cooperative
queue: users with direct NVIDIA device access can bypass it. Device enforcement
requires an administrator-managed cgroup/container policy and is not implemented
by this package.

## Commands

```bash
gpuq status
gpuq ls
gpuq run -g 1 -- python train.py --config experiment.yaml
gpuq submit -g 2 -- torchrun --standalone --nproc_per_node=2 train.py
gpuq logs -f JOB_ID
gpuq cancel JOB_ID
gpuq show JOB_ID
```

`run` and `submit` release cards only after the command AND its descendants exit.
When the command exits, remaining descendants are terminated, first with SIGTERM
and then SIGKILL. Do not submit launcher scripts that intentionally background
the real workload and exit; keep the training command in the foreground.

The workload must remain in the supervisor's descendant tree. Launching work
through an existing Docker daemon, tmux server, or `systemd-run` is unsupported:
those services create processes outside that tree. Starting `gpuq run` INSIDE
an existing tmux terminal is supported. Linux process adoption uses
[PR_SET_CHILD_SUBREAPER](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html).

Allocation uses GPU UUIDs in `CUDA_VISIBLE_DEVICES`. Inside a two-card job, use
logical `cuda:0` and `cuda:1`. Do not override the variable or hard-code physical
indices. Conda/PATH, working directory, and file ownership remain user-owned.
Supervisor diagnostics and training output share the job log. Foreground output
is mirrored from that file independently of heartbeats.

## Existing training commands in a shell

```bash
gpuq shell -g 1
conda activate my-environment
python train.py --config experiment.yaml
exit
```

The shell reserves the card until exit. By default it closes after 1800 seconds
with no child commands. This measures absence of child processes, not keyboard
activity or GPU utilization; shell builtins alone do not reset the timer.
Set `--idle-timeout 600` to shorten it or `--idle-timeout 0` to disable it.
Exiting/cancelling the shell also cleans up its child jobs, including background
workers and independent process groups. For automatic release at training end,
prefer `gpuq run` or `gpuq submit`.

## Failure behavior

- State transitions use SQLite write transactions; allocations cannot overwrite
  concurrent cancellation/expiry decisions.
- Unstarted pending jobs without a supervisor heartbeat expire to `lost`.
  Allocation waits for supervisor registration, so a failed background launch
  cannot reserve a GPU.
- Allocated/running jobs without a heartbeat become `quarantined` (`Q`), retaining
  every allocated card. CUDA-process absence alone never releases such a lease.
- A returning supervisor cleans up a quarantined task and then confirms finish.
- On daemon downtime, an existing running job continues. Completion notification
  is retried until the daemon returns using the SAME state directory.
- Linux child-subreaper tracking retains orphaned workers, even after `setsid`.
  Cancellation escalates while descendants survive; only an empty task tree
  permits release. A task stuck in uninterruptible kernel sleep keeps its lease.
- If the supervisor is killed with SIGKILL, automatic cleanup cannot be guaranteed.
  The lease stays quarantined. There is deliberately no timeout-based force-release
  command. Operators must investigate and terminate residual processes; this version
  does not yet provide automated reconciliation for a permanently dead supervisor.
- GPU UUID/index topology changes with active leases prevent daemon startup.
  Drain jobs before changing hardware or migrating state; do not delete state to
  bypass this check.

## Scheduling

Default `gpuqd --policy fifo` preserves submission order. A four-card head job
can hold up smaller jobs while only three cards are free.

`gpuqd --policy fit` skips currently unsatisfiable jobs and starts ones that fit.
This improves utilization but may starve large jobs; it is not time-reserved
backfilling. No quotas, priorities, preemption, or GPU sharing are implemented.

## Manual demo (one daemon)

Choose a daemon-owned directory, not a socket directly in shared `/tmp`:

```bash
mkdir -p "$HOME/.local/run/gpuq"
chmod 755 "$HOME/.local/run/gpuq"
export GPUQ_SOCKET="$HOME/.local/run/gpuq/gpuq.sock"
export GPUQ_DAEMON_UID="$(id -u)"
gpuqd --socket "$GPUQ_SOCKET" --state-dir "$HOME/.local/state/gpuqd"
```

In each client terminal set the same socket path and the daemon's explicit UID.
Clients default to trusting UID 0; they check both socket ownership and kernel
peer credentials before sending requests. Commands, cwd and log destinations
stay in the submitting client instead of being taken from daemon replies.

## Server-wide installation (administrator)

First drain and stop ALL legacy demo daemons. The new per-GPU kernel lock blocks
two upgraded daemons even with different databases, but cannot coordinate with
an older daemon that did not implement locking. Do not overwrite modules while
live legacy supervisors are still using them.

From a reviewed checkout, with the old daemon stopped and no active jobs:

```bash
sudo mkdir -p /opt/gpuq/src
sudo cp -r src/gpuq /opt/gpuq/src/
sudo install -m 0755 deploy/gpuq /usr/local/bin/gpuq
sudo install -m 0755 deploy/gpuqd /usr/local/bin/gpuqd
sudo install -m 0644 deploy/gpuqd.service /etc/systemd/system/gpuqd.service
sudo systemctl daemon-reload
sudo systemctl enable --now gpuqd
gpuq status
journalctl -u gpuqd
```

The service uses `/run/gpuq/gpuq.sock` and `/var/lib/gpuqd`. For a fresh system
service, remove old demo `GPUQ_SOCKET`/`GPUQ_DAEMON_UID` overrides in client shells.
The daemon must remain in the host PID/network namespace. An administrator can
restrict membership/access to the socket separately; the daemon does not execute
submitted commands as root.

## Verification

```bash
python -m pytest -q
```

Lifecycle integration tests require Linux and use simulated GPUs, not training
hardware. macOS runs state/security/client tests only. They cover transactional
expiry, quarantined leases, TERM-resistant and session-detached descendants,
supervisor suspension, daemon restart, inherited stdout, authentication, topology
checks and per-GPU instance locks. Real multi-user device isolation remains a
separate deployment requirement.
