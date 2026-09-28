GTX 1070 extraction host -- setup runbook (T5, goal mist-two-loop / MIS-171)
===========================================================================

Who runs this: Raj and the lead, on the host. Nothing here is run by a delegate.
A real build has run most of this runbook (see the "Measured on the real host
build" / "VERIFIED" notes throughout), including a re-fit at the current
context size (16384) -- see step 5.3 and `docker/extraction/README.md`'s
"Host ncmoe sizing" table. Every step still marked [UNVERIFIED] is a reasoned
instruction that has not been checked on the real machine; replace the marker
with what was observed when you run it.

What the host is for: it runs the standalone compose file
`docker/extraction/compose.host.yml` -- ONLY the stateless extraction service
(`mist-extraction-host`), its own llama-server (`mist-extraction-llm-host`), and a
Tailscale sidecar (`mist-extraction-ts`). This file needs no other compose file to
resolve, and in particular it never starts the repo root `docker-compose.yml`'s main
stack (mist-llm, mist-neo4j, mist-backend) -- that stack has no business running on
this machine. The MIST backend on the main machine reaches this host over the
tailnet and is the only client. The host holds no graph, no event store and no MIST
address: it can be wiped and rebuilt at any time, and a backlog simply accumulates
on the main machine while it is down.

Host RAM, CPU and OS, needed before step 4: Win10 Home 22H2, WSL2 Ubuntu
26.04, Docker Engine 29.8.1 (not Docker Desktop), nvidia-container-toolkit
1.20.1, per the real host build. The expert-offload setting and prompt
throughput depend on these; if you are setting up a DIFFERENT machine, confirm
its own RAM/CPU/OS before step 4 rather than assuming this one's.

---

1. Driver (Pascal, sm_61)
-------------------------

The pinned llama.cpp image is `server-cuda-b11151` built on CUDA 12.8.1. Its
default architecture list compiles sm_61 as PTX only, so the driver JIT-compiles
the kernels on first load (source: lead's b11151 check,
`ggml/src/ggml-cuda/CMakeLists.txt:26-32` in llama.cpp at tag b11151).

- Install the NVIDIA R580 driver branch. It supports Pascal and satisfies CUDA
  12.8's minimum (570.65 or later on Windows per NVIDIA's release notes). Do NOT
  install a CUDA 13 driver-only stack that drops Pascal.
- Check: `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`
  should report `GeForce GTX 1070`, a 580.x driver and compute capability `6.1`.
  [UNVERIFIED]
- Linux host: install the NVIDIA Container Toolkit so compose's
  `deploy.resources.reservations.devices` reaches the GPU. Windows host: either
  Docker Desktop with the WSL2 backend, OR Docker Engine (not Docker Desktop)
  installed directly inside WSL2 Ubuntu with nvidia-container-toolkit -- both
  are valid; the real host build used the latter (Win10 Home 22H2, WSL2 Ubuntu
  26.04, Docker Engine 29.8.1, nvidia-container-toolkit 1.20.1).

1a. WSL keepalive (Linux-host-in-WSL setups only)
--------------------------------------------------

WSL 2.7.14 tears down the distro when the last `wsl.exe` client exits, even
with systemd enabled inside the distro -- `vmIdleTimeout` does not prevent
this. The fix is a persistent keepalive WSL session, so the distro (and the
Docker Engine/containers running in it) survives without an interactive
`wsl.exe` session attached. This is only relevant to a Docker-Engine-in-WSL
setup, not Docker Desktop.

The original implementation was a simple at-logon scheduled task holding a
`sleep infinity` process alive inside the distro. That task has since been
killed by Windows three times (error `0xC000013A`). It has been rebuilt to
be self-healing: a Windows Scheduled Task with a `TimeTrigger` that
re-checks and, if needed, restarts the keepalive session every 5 minutes,
rather than a one-shot task that only runs at logon. Worst-case downtime if
the keepalive process dies is therefore about 5 minutes (until the next
trigger check) plus about 30 seconds for WSL and the containers inside it
to come back up.

**Other benign quirks observed on the real host build, noted here only where
they generalize beyond that one machine:** `sudo` inside the WSL distro may
prompt for a password -- use `wsl -u root` (run the command directly as root
inside the distro, from the Windows host) as an alternative where `sudo`
inside the distro is unavailable or blocks. This is a WSL-invocation
workaround, not a substitute for `sudo` on a Linux host. The NVIDIA driver
inside WSL can log benign `dxgk` ioctl `-75` noise; this can be ignored.

2. Images, pulled by digest
---------------------------

Pull exactly the pinned references, so the host runs the same build the
benchmarks measured:

    docker pull ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:014f721265464f38ccb247c1338d07d852c4bae7509a4b4734d07a2bbadc765c
    docker pull tailscale/tailscale@sha256:c507f3a2a6ab1cabd8d809b98edeb41edbd5c3fb6ad9632ffd098b4c7d0b4065

The service image is built from `docker/extraction/Dockerfile` (its base is pinned
to `python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e`).
Build it on the main machine and transfer it, or build it on the host from a
checkout of the same commit:

    docker compose -f docker/extraction/compose.host.yml build

3. Model file
-------------

Copy `ggml-org/gpt-oss-20b-MXFP4.gguf` (11.28 GiB, 12.1 GB decimal, the file
`scripts/model_bench`
arm `c9` benchmarked) into the host's models directory at the same relative path
the compose file mounts (see `docker/extraction/README.md`, section "Model").
Record its sha256; the value you choose for `EXTRACTION_MODEL_HASH` names this
exact file, and it defines the extraction epoch.

4. Environment
--------------

Create `.env` beside `docker/extraction/compose.host.yml` on the host (never commit
it) -- compose loads `.env` from the directory of the compose file you pass with
`-f`, not the invoking shell's working directory:

    EXTRACTION_MODEL_HASH=<the epoch name for this file, e.g. gpt-oss-20b-mxfp4-<sha8>>
    TS_AUTHKEY=<a tailnet auth key, see step 6>
    EXTRACTION_HOST_NCMOE=24       # all experts on CPU (verified block_count=24
                                    # via GGUF metadata); step 5 measures a tighter fit
    MODELS_DIR=<absolute host path to the models directory>  # overrides the
                                    # ../../models default, which will not exist on
                                    # a fresh host

Every other `EXTRACTION_*` variable has a default; `docker/extraction/README.md`
section "Environment variables" lists them.

**Out-of-repo timeout override, retirement.** Before this compose passthrough
landed, `compose.host.yml` had no way to forward `EXTRACTION_LLM_TIMEOUT_SECONDS`
at all, so the live host set it to `300` via an out-of-repo override file
(`/home/user/overrides/timeout.yml` on the host, loaded by the host's own
`start-mist-stack.cmd` script) as a workaround. Now that
`EXTRACTION_LLM_TIMEOUT_SECONDS` is forwarded (see
`docker/extraction/compose.host.yml`'s `mist-extraction-host` service), that
override is no longer load-bearing. On the host:

1. Set `EXTRACTION_LLM_TIMEOUT_SECONDS=300` (or whatever value is in use)
   directly in the host's own `.env` beside `docker/extraction/compose.host.yml`
   (the same file created in this step, next to `EXTRACTION_MODEL_HASH` and
   `TS_AUTHKEY`).
2. Delete `/home/user/overrides/timeout.yml`.
3. Remove `start-mist-stack.cmd`'s reference to that override file.

`start-mist-stack.cmd` is out-of-repo and host-local -- these three steps are
for Raj or the operator to carry out on the host, not something a delegate
can do from this checkout.

5. First start, Pascal check and fit measurement
------------------------------------------------

5.1 First start (PTX JIT). Start only the llama-server first and time it:

    docker compose -f docker/extraction/compose.host.yml up -d mist-extraction-llm-host
    docker compose -f docker/extraction/compose.host.yml logs -f mist-extraction-llm-host

The first start JIT-compiles every kernel. The compiled cache lands on the
named volume `mist-extraction-cuda-cache` (`CUDA_CACHE_PATH`), so a second
start should be much faster. Record both times on your own hardware -- the
real host build's numbers below are one data point, not a guaranteed bound.
If the second start is as slow as the first, the cache is not persisting:
check the volume mount before going further.

**Measured on the real host build:** 64 s cold (first start, uncached), 36 s
warm (second start) -- the CUDA cache volume persisting across restarts is what
makes the second start faster.

**Volume naming note:** running `compose.host.yml` without an explicit project
name derives the project name from the compose file's own directory
(`extraction`), and Compose prefixes every named volume with it -- the volume
actually created on the host is `extraction_mist-extraction-cuda-cache`, not
the bare `mist-extraction-cuda-cache` written in the compose file. Check
`docker volume ls` on the host for the prefixed name.

5.2 Service check. b11151's llama-server log prints no CUDA device line and no
offload lines at all for this build -- do not look for them; there is nothing
observable there to confirm. Verify instead via the endpoints the host actually
serves:

    docker compose -f docker/extraction/compose.host.yml exec mist-extraction-llm-host curl -s localhost:8080/health

must return status ok. (curl is in the server image: the compose healthcheck
for this service and for `mist-llm` both call it.) Then, once the extraction
service is also up, confirm `/v1/info` (see step 6) reports the expected
contract version, extraction version, model hash, adapter, build, and location
label -- the real host build's `/v1/info` reported contract `1.0.0`, extraction
`2026-06-14-r5`, model `gpt-oss-20b-mxfp4-27cd6c43`, `b11151`, adapter
`gptoss`, label `host-1070`. `/health` and `/v1/info` returning ok do not by
themselves confirm the GPU is in use (a CPU-only fallback would also answer
them) -- confirm actual GPU use via `nvidia-smi` memory consumption during a
request (step 5.3). The GGUF file's own metadata is the source for the
model's total MoE layer count (`block_count`, used in step 5.3's sweep), not
for what the GPU loaded at runtime -- b11151's load log states neither.

5.3 Fit. The 1070 has 8 GB. gpt-oss-20b measured about 10.8 GiB on the 4070 with
`ncmoe=6` (plan v1), so the host needs most or all expert layers on the CPU.
gpt-oss-20b's MoE layer count is VERIFIED as 24 via the GGUF file's own
`block_count` metadata (b11151's llama-server load log is not confirmed to
print this -- the real host build found it prints no CUDA device or offload
lines at all; read the layer count from the GGUF file itself rather than
relying on the log). Sweep `EXTRACTION_HOST_NCMOE`
downward from 24 (fewer experts on CPU means more VRAM and faster decode),
restarting the llama-server each time, and record for each value:
- peak VRAM from `nvidia-smi --query-gpu=memory.used --format=csv -l 1` during a
  request;
- decode tokens/s and prompt tokens/s from the llama-server timings;
- host RAM in use and swap activity (experts on CPU live in system RAM; RAM/swap
  pressure is WORST at HIGH `ncmoe`, not low -- more experts held on CPU means
  more system RAM used).
Keep at least 512 MiB of VRAM headroom, the margin the `scripts/model_bench`
fit clause uses. Pick the lowest value that fits, write it into `.env`, and
record the sweep in the PR or the vault note.

**Re-fit at ctx 16384** (9,272-token prompt, `max_tokens=2048`), the current
ctx-size default:

| `ncmoe` | peak VRAM | headroom | prompt t/s | decode t/s | `/v1/extract` |
|---|---|---|---|---|---|
| 12 | 7632 MiB | 560 MiB | 499 | 17.7 | 51.3 s |
| 13 | 7244 MiB | 948 MiB | 483 | 16.4 | 51.5 s |
| 14 | 6840 MiB | 1352 MiB | 481 | 14.9 | 51.8 s |

No swap at any of these values. Idle VRAM is about 7560 MiB at `ncmoe=12`.
The prompt cache works. Entity count varies slightly across `ncmoe` (4 vs 3
entities on a fixed test conversation); relationships are identical. Prompt
throughput across the sweep is approximately 500 t/s (the earlier recorded
~155 t/s figure was measured at the old ctx-8192/406-token setup and is
superseded by this table).

**Recommended `ncmoe` for this host: 13, not the bare-minimum 12.** Step
5.3's general "pick the lowest value that fits" guidance above assumes VRAM
is otherwise idle; this GPU also drives the host's own display, so `ncmoe=12`
leaves only 560 MiB headroom -- tighter than is comfortable when the same
card is also rendering a desktop. `ncmoe=13` leaves 948 MiB headroom for a
small, predictable decode-speed cost (16.4 vs 17.7 t/s). Set
`EXTRACTION_HOST_NCMOE=13` in the host's own `.env` (`docker/extraction/compose.host.yml`'s
own committed default stays `24`, all-experts-on-CPU, until an operator
opts into a tighter value this way).

**Recommended operator setting: `ncmoe=13`, not the bare minimum `ncmoe=12`.**
This GPU also drives the host's own display -- `12` fits with only 560 MiB
spare, which is tighter than desirable when the same card is also rendering
a desktop. `13` leaves substantially more headroom (948 MiB) for a small,
predictable decode-speed cost (16.4 vs 17.7 t/s). Write
`EXTRACTION_HOST_NCMOE=13` into `.env` to opt in; `compose.host.yml`'s own
committed default stays `24` (all experts on CPU) until an operator does.

6. Tailscale exposure
---------------------

The host profile publishes no ports. The service shares the sidecar's network
namespace, so it is reachable only on the tailnet, at the sidecar's hostname,
port 8090. The llama-server is reachable only inside the compose network.

- Create a tagged, non-ephemeral auth key in the tailnet admin console and put
  it in `TS_AUTHKEY`. Use an ACL that allows ONLY the main MIST machine to reach
  this host on tcp/8090.

  **Union-semantics caveat, confirmed on the real host build:** Tailscale ACL
  grants are additive (a connection is allowed if ANY rule permits it), so a
  pre-existing catch-all allow-all grant makes a narrower, host-specific rule
  a no-op -- the narrow rule adds a permission but the catch-all grant already
  allows everything, so nothing is actually restricted until the catch-all
  grant itself is narrowed. The real build narrowed the catch-all grant to
  `dst autogroup:member` before adding the host-specific rule. The applied,
  Raj-approved policy on the real host build: the catch-all grant narrowed to
  `dst autogroup:member`, plus a rule allowing only `<main MIST machine> ->
  tag:mist-extraction tcp:8090`.

  **If scripting this via the Tailscale API instead of the admin console:**
  the real host build's session found the ACL/key update call needs POST, not
  PUT; the key description field has a length limit; and conditional updates
  need an `If-Match` header. This is a brief note for anyone who scripts it --
  the admin console is the documented path above, no API-scripted flow ships
  in this repo, and the exact endpoint/call was not recorded.
- Start the rest of the profile:

      docker compose -f docker/extraction/compose.host.yml up -d

- From the MIST machine, check the contract endpoints:

      # TS_HOSTNAME defaults to mist-extraction-gtx1070 in the compose file
      curl -s http://<TS_HOSTNAME>:8090/v1/health
      curl -s http://<TS_HOSTNAME>:8090/v1/info

  `/v1/info` must report `extraction_version`, `model_hash` (your
  `EXTRACTION_MODEL_HASH`), `llama_cpp_build` `b11151` and `location_label`
  `host-1070`.
- On the MIST machine, set `MIST_EXTRACTION_SERVICE_URL=http://<TS_HOSTNAME>:8090`
  for the backend and restart it. **Verified reachable:** the lead confirmed
  `mist-extraction-gtx1070:8090/v1/health` and the tailnet IP's
  `:8090/v1/health` both answer from inside the `mist-backend` container, so
  the main backend container already reaches the tailnet with no additional
  networking change needed.

7. Wake-on-LAN (manual, Raj's decision 5)
-----------------------------------------

The host may sleep. While it does, the backend's dispatcher reports state
`unreachable` in `extraction_status`, counts no failed attempts, and the backlog
accumulates in log order. When the host wakes, the backlog drains on its own.

- Enable Wake-on-LAN in the host's BIOS/UEFI and on its NIC (Windows: the
  adapter's Power Management tab, "Allow this device to wake the computer" and
  "Only allow a magic packet"). [UNVERIFIED on this machine]
- Record the host NIC's MAC address.
- A magic packet is a LAN broadcast: send it from a machine on the SAME LAN, not
  over Tailscale. From the MIST machine (Windows PowerShell), send the standard
  magic packet (6 bytes of 0xFF, then the MAC repeated 16 times) as UDP to the
  LAN broadcast address on port 9, with any Wake-on-LAN utility you trust.
  [UNVERIFIED: tool choice; no script ships in this repository]
- A host-side waker that sends the packet automatically when the backlog grows is
  the later step Raj chose; it is not built.

8. Smoke test
-------------

With the host up and the backend pointed at it:

    curl -s http://localhost:8001/extraction/status

should show `state` `idle` or `working`, `service.reachable` true, and
`service.location_label` `host-1070`. Send one text turn through MIST, then watch
`backlog_depth` return to 0 and `last_job.outcome` become `applied`.

If `state` is `epoch_mismatch`, the host's model does not match the active epoch.
That is expected the first time the extraction model changes. Switching models is
an epoch cutover: follow `backend/extraction_backlog/CUTOVER.md`, not an env edit
on the backend.

9. Record
---------

Record in the PR or the vault note: driver version, first and second start times,
the ncmoe sweep and the chosen value, decode and prompt tokens/s, the tailnet
hostname, and the WoL MAC and tool.
