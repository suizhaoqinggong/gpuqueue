# gpuq

gpuq 是单机 NVIDIA GPU 协作式任务队列。`gpuqd` 分配 GPU 租约，用户侧的
supervisor 以提交者自己的身份启动命令，运行时仅依赖 Python 标准库。

实际部署需要 Linux、NVIDIA GPU 和 Python 3.9+；macOS 可运行状态、安全和客户端测试。
这是协作式队列：拥有 GPU 设备访问权限的用户仍可绕过队列，设备级隔离需要管理员另行配置。

## 安装与快速使用

```bash
pip install -e '.[dev]'

# 以下命令要求管理员已启动 gpuqd
gpuq status
gpuq run -g 1 -n experiment -- python train.py --config experiment.yaml
gpuq submit -g 2 -n distributed -- torchrun --standalone --nproc_per_node=2 train.py
gpuq ls
gpuq logs -f 42
```

示例中的 `42` 是任务 ID，请替换为提交时输出的实际 ID。首次部署和 systemd
安装步骤见 [部署文档](docs/gpuq.md#server-wide-installation-administrator)。

## 命令速查

| 命令 | 用途 |
| --- | --- |
| `gpuq run` | 排队并在前台等待命令完成，实时显示输出 |
| `gpuq submit` | 提交后台任务，提交后立即返回 |
| `gpuq shell` | 排队获取 GPU 后打开交互式 Shell |
| `gpuq ls` / `gpuq list` | 查看任务队列与历史记录 |
| `gpuq status` | 查看物理 GPU 的分配状态 |
| `gpuq show JOB_ID` | 查看单个任务的完整 JSON 信息 |
| `gpuq cancel JOB_ID` | 请求取消任务 |
| `gpuq logs JOB_ID` | 查看任务日志 |
| `gpuqd` | 启动守护进程或执行离线恢复 |

### 通用参数与帮助

```bash
gpuq --help
gpuq run --help
gpuqd --help
gpuq --socket /run/gpuq/gpuq.sock status
```

- `-h` / `--help`：显示当前命令的帮助并退出。
- `gpuq --socket PATH`：指定 Unix socket，默认取 `GPUQ_SOCKET`，未设置时为
  `/run/gpuq/gpuq.sock`。这个全局选项应放在子命令之前。
- `run`、`submit` 和 `shell` 共用 `-g N` / `--gpus N`（默认 `1`）和
  `-n NAME` / `--name NAME`（默认取执行程序的文件名）。GPU 数量必须在
  `1` 到守护进程管理的 GPU 总数之间；这些选项应放在执行命令之前。
- `run`、`submit` 使用 `--` 分隔 gpuq 参数与工作负载命令，后面的参数原样传给程序。

### `gpuq run`：前台运行

```bash
gpuq run [-g N] [-n NAME] -- COMMAND [ARG ...]

gpuq run -g 1 -n baseline -- python train.py --epochs 100
gpuq run -g 2 -- torchrun --standalone --nproc_per_node=2 train.py
# 管道、重定向等 Shell 语法需要显式调用 Shell
gpuq run -g 1 -- bash -lc 'python train.py && python evaluate.py'
```

提交后保持前台等待，有可用 GPU 时启动命令，并将合并后的标准输出、标准错误写入日志、
实时显示在终端。工作负载的标准输入关闭，需要交互输入时请使用 `gpuq shell`。
`Ctrl+C` 请求取消任务。正常结束时返回工作负载的退出码。

命令继承当前环境和提交时的工作目录，可先激活 Conda 环境再运行。前台 supervisor
需要持续运行；长时间训练可在 tmux 终端内执行 `gpuq run`，或使用 `gpuq submit`。

### `gpuq submit`：后台提交

```bash
gpuq submit [-g N] [-n NAME] -- COMMAND [ARG ...]

gpuq submit -g 1 -n overnight -- python train.py --epochs 200
gpuq ls
gpuq logs -f 42
gpuq show 42
```

启动独立的用户侧 supervisor 后立即返回，输出任务 ID 和日志路径。命令在后台排队、
获得 GPU 后执行；退出码 `0` 表示提交成功，训练是否成功需要通过 `ls`、`show` 或日志查看。
该模式同样不提供交互式标准输入。不要在启动脚本中把训练放到后台后立即退出；
主命令结束时 gpuq 会清理剩余后代进程。

### `gpuq shell`：交互式使用 GPU

```bash
gpuq shell [-g N] [-n NAME] [--shell PATH] [--idle-timeout SECONDS]

gpuq shell -g 1 -n debug --shell /bin/bash --idle-timeout 600
# 获得 GPU、进入 Shell 后执行：
conda activate my-environment
python train.py
exit
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--shell PATH` | `$SHELL`，未设置则 `/bin/bash` | 用指定程序打开登录 Shell（传入 `-l`） |
| `--idle-timeout SECONDS` | `1800` | 无子进程达到指定秒数后关闭 Shell；`0` 禁用，不能为负数 |

需要交互式终端。GPU 会保留到 Shell 退出，单次训练结束不会立即释放。
空闲计时依据是否存在子进程，不依据键盘操作或 GPU 利用率；仅执行 Shell 内建命令
不会重置计时。退出或取消 Shell 会清理其后代进程，包括后台工作进程。
Shell 会话不采集日志，不能用 `gpuq logs` 回看。

### `gpuq ls` / `gpuq list`：查看队列与历史

```bash
gpuq ls [--all] [--history] [--limit N] [--before JOB_ID]

gpuq ls                                  # 自己的所有活跃任务及最近的完成任务
gpuq list --all                          # 所有用户的任务摘要
gpuq ls --history --limit 50              # 仅查看最近 50 条完成记录
gpuq ls --history --before 100 --limit 50 # 查看 ID 小于 100 的完成记录
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--all` | 关闭 | 包含其他用户的任务摘要，不授予查看详情或取消权限 |
| `--history` | 关闭 | 仅显示已完成任务 |
| `--limit N` | `200` | 历史记录每页条数，范围 `1..200`；不限制活跃任务数量 |
| `--before JOB_ID` | 无 | 只取 ID 小于指定正整数的历史记录，不影响活跃任务 |

默认显示全部活跃任务（包含隔离任务）和最多 200 条最近完成记录。有更多历史记录时，
输出末尾会给出下一页命令。摘要包含任务 ID、状态、用户、GPU、提交至今时长和名称；
`GPU` 列中的 `2 req` 表示申请两张卡，数字列表则表示已分配的物理 GPU 编号。

| 状态 | 含义 |
| --- | --- |
| `PD` | pending，排队中 |
| `AL` | allocated，已分配，等待启动 |
| `R` | running，运行中 |
| `CD` | succeeded，成功完成 |
| `F` | failed，执行失败 |
| `CA` | cancelled，已取消 |
| `LO` | lost，任务丢失，例如未启动任务心跳超时 |
| `Q` | quarantined，租约隔离，仍占用已分配 GPU |

### `gpuq status`：查看 GPU 状态

```bash
gpuq status
```

无专用参数。显示物理 GPU 编号、状态、关联任务 ID、总显存（MiB）及型号。
状态为 `idle`（空闲）、`allocated`（已有队列租约）、`external`（队列外进程占用）
或 `quarantine`（隔离租约）。显存列是总容量，不是实时已用显存。

### `gpuq show`：查看任务详情

```bash
gpuq show JOB_ID
gpuq show 42
```

以 JSON 输出任务详情，例如命令、工作目录、日志路径、GPU 分配、时间和退出码。
仅任务所有者和管理员可查看；管理员是运行 `gpuqd` 的用户。

### `gpuq cancel`：取消任务与释放隔离租约

```bash
gpuq cancel JOB_ID [--force --reason TEXT]
gpuq cancel 42
```

普通取消仅允许任务所有者或管理员操作。运行中的任务由 supervisor 清理进程，
确认整个任务进程树退出后才释放 GPU；命令返回只表示取消请求已提交。

`--force` 仅供管理员手动释放 `Q` 状态租约，必须同时提供非空 `--reason`。
管理员需先检查并清理残留进程，再执行：

```bash
gpuq cancel 42 --force --reason "verified and cleaned up all residual processes"
```

强制释放会把任务标记为 `cancelled`，本身不负责终止残留进程。异常处理详见
[故障行为说明](docs/gpuq.md#failure-behavior)。

### `gpuq logs`：查看日志

```bash
gpuq logs JOB_ID [-n N] [-f]

gpuq logs 42                # 最后 100 行
gpuq logs 42 --lines 200     # 最后 200 行
gpuq logs 42 --follow       # 显示最后 100 行并持续追踪
gpuq logs 42 -n 0 -f        # 仅追踪后续新增内容
```

- `-n N` / `--lines N`：显示末尾行数，默认 `100`，必须为非负整数。
- `-f` / `--follow`：持续追踪日志，用 `Ctrl+C` 停止查看，不会取消任务。

日志包含工作负载输出，也可能包含 supervisor 诊断信息。日志位于提交用户的本地文件系统，
查看时需要具备任务详情权限及日志文件读取权限；此命令使用系统 `tail`。
任务完成后 `--follow` 仍会等待新输出，需要手动退出。

### `gpuqd`：启动守护进程（管理员）

```bash
gpuqd [--socket PATH] [--state-dir PATH] [--interval SECONDS] \
      [--stale-after SECONDS] [--policy {fifo,fit}]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--socket PATH` | `$GPUQ_SOCKET` 或 `/run/gpuq/gpuq.sock` | 客户端连接的 Unix socket |
| `--state-dir PATH` | `$GPUQ_STATE_DIR` 或 `~/.local/state/gpuqd` | 数据库、锁及 GPU 拓扑的持久化目录 |
| `--interval SECONDS` | `1.0` | 调度循环间隔，必须大于零 |
| `--stale-after SECONDS` | `30.0` | 心跳超时阈值，必须大于零 |
| `--policy {fifo,fit}` | `fifo` | `fifo` 按提交顺序等待；`fit` 跳过暂时无法满足的请求，可能使大任务长期等待 |
| `--recover-job JOB_ID` | 无 | 离线释放隔离租约；必须先停止守护进程 |
| `--reason TEXT` | 空 | 离线恢复原因；使用 `--recover-job` 时必须非空 |

单用户手动启动示例（Linux NVIDIA 主机）：

```bash
mkdir -p "$HOME/.local/run/gpuq"
chmod 755 "$HOME/.local/run/gpuq"
export GPUQ_SOCKET="$HOME/.local/run/gpuq/gpuq.sock"
export GPUQ_DAEMON_UID="$(id -u)"
gpuqd --socket "$GPUQ_SOCKET" --state-dir "$HOME/.local/state/gpuqd" --policy fifo
```

在其他客户端终端设置相同的 `GPUQ_SOCKET` 和 `GPUQ_DAEMON_UID` 后即可提交任务。
Socket 的所有父目录必须由 root 或守护进程用户所有，且不能允许组或其他用户写入；
不要直接使用共享 `/tmp` 下的 socket。守护进程应运行在宿主机 PID/网络命名空间。

离线恢复需以原守护进程用户操作，使用原状态目录，确认残留进程已清理后执行：

```bash
# 先停止 gpuqd；系统服务默认使用 /var/lib/gpuqd
gpuqd --state-dir /var/lib/gpuqd --recover-job 42 \
       --reason "manual offline release after verification"
```

### 内部命令 `_supervise`

`gpuq _supervise` 由 `gpuq submit` 自动启动，通过标准输入接收内部 JSON 配置，
负责后台任务生命周期。无需手动调用。

## 环境变量与运行约定

| 环境变量 | 用途与默认值 |
| --- | --- |
| `GPUQ_SOCKET` | 客户端和守护进程 socket 默认路径，默认 `/run/gpuq/gpuq.sock` |
| `GPUQ_DAEMON_UID` | 客户端信任的守护进程 UID，默认 `0`；非 root 部署时显式设置 |
| `GPUQ_STATE_DIR` | 守护进程状态目录，默认 `~/.local/state/gpuqd`；systemd 服务使用 `/var/lib/gpuqd` |
| `GPUQ_CLIENT_STATE_DIR` | 用户侧状态目录，默认 `~/.local/state/gpuq`；日志存于其 `logs/` 子目录 |
| `SHELL` | `gpuq shell` 未指定 `--shell` 时使用的程序 |

工作负载会收到 gpuq 设置的 `CUDA_VISIBLE_DEVICES`（GPU UUID 列表）、
`CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `GPUQ_JOB_ID`。申请两张卡时，程序内使用逻辑
`cuda:0`、`cuda:1`；不要覆盖设备可见性变量或硬编码物理卡编号。

任务必须保持在 supervisor 的后代进程树中。通过已有 Docker daemon、tmux server
或 `systemd-run` 启动工作负载不受支持；在已有 tmux 终端内运行 `gpuq run` 则受支持。
守护进程暂时离线时，已运行任务会继续，完成通知会重试；恢复时应保留原状态目录。
心跳超时的已分配任务会进入隔离状态，不会仅因 GPU 暂无 CUDA 进程就自动释放租约。

## 项目结构与验证

```text
src/gpuq/     队列、客户端和守护进程实现
tests/        pytest 测试
deploy/       客户端包装脚本、守护进程包装脚本和 systemd unit
docs/gpuq.md  调度策略、故障行为与服务器部署说明
```

```bash
python -m pytest -q
```

生命周期集成测试需要 Linux，使用模拟 GPU，无需训练硬件。macOS 可运行状态、安全和客户端测试。
