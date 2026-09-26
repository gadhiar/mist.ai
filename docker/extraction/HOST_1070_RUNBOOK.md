GTX 1070 extraction host -- setup runbook (T5, goal mist-two-loop / MIS-171)
===========================================================================

Who runs this: Raj and the lead, on the host. Nothing here is run by a delegate,
and nothing here has been executed yet. Every step marked [UNVERIFIED] is a
reasoned instruction that has not been checked on the real machine; replace the
marker with what was observed when you run it.

What the host is for: it runs the `extraction-host` profile of
`docker-compose.extraction.yml` -- the stateless extraction service
(`mist-extraction-host`), its own llama-server (`mist-extraction-llm-host`), and a
Tailscale sidecar (`mist-extraction-ts`). The MIST backend on the main machine
reaches it over the tailnet and is the only client. The host holds no graph, no
event store and no MIST address: it can be wiped and rebuilt at any time, and a
backlog simply accumulates on the main machine while it is down.

Open inputs, needed before step 4:
- Host RAM, CPU and OS (Raj's specs). The expert-offload setting and prompt
  throughput depend on them.

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
  `deploy.resources.reservations.devices` reaches the GPU. Windows host: Docker
  Desktop with the WSL2 backend. [UNVERIFIED which OS the host runs]

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

    docker compose -f docker-compose.extraction.yml --profile extraction-host build

3. Model file
-------------

Copy `ggml-org/gpt-oss-20b-MXFP4.gguf` (12.1 GB, the file `scripts/model_bench`
arm `c9` benchmarked) into the host's models directory at the same relative path
the compose file mounts (see `docker/extraction/README.md`, section "Model").
Record its sha256; the value you choose for `EXTRACTION_MODEL_HASH` names this
exact file, and it defines the extraction epoch.

4. Environment
--------------

Create `.env` beside the compose file on the host (never commit it):

    EXTRACTION_MODEL_HASH=<the epoch name for this file, e.g. gpt-oss-20b-mxfp4-<sha8>>
    TS_AUTHKEY=<a tailnet auth key, see step 6>
    EXTRACTION_HOST_NCMOE=999      # provisional: all experts on CPU; step 5 measures

Every other `EXTRACTION_*` variable has a default; `docker/extraction/README.md`
section "Environment variables" lists them.

5. First start, Pascal check and fit measurement
------------------------------------------------

5.1 First start (PTX JIT). Start only the llama-server first and time it:

    docker compose -f docker-compose.extraction.yml --profile extraction-host up -d mist-extraction-llm-host
    docker compose -f docker-compose.extraction.yml --profile extraction-host logs -f mist-extraction-llm-host

The first start JIT-compiles every kernel and can take several minutes. The
compiled cache lands on the named volume `mist-extraction-cuda-cache`
(`CUDA_CACHE_PATH`), so a second start should be much faster. Record both times.
If the second start is as slow as the first, the cache is not persisting: check
the volume mount before going further. [UNVERIFIED: first-start duration]

5.2 Pascal check. The llama-server log must list the GTX 1070 as a CUDA device
with compute capability 6.1 and offload layers to it. Then:

    docker compose -f docker-compose.extraction.yml --profile extraction-host exec mist-extraction-llm-host curl -s localhost:8080/health

must return status ok. (curl is in the server image: the compose healthcheck
for this service and for `mist-llm` both call it.)

5.3 Fit. The 1070 has 8 GB. gpt-oss-20b measured about 10.8 GiB on the 4070 with
`ncmoe=6` (plan v1), so the host needs most or all expert layers on the CPU. Sweep
`EXTRACTION_HOST_NCMOE` downward from 999 (fewer experts on CPU means more VRAM
and faster decode), restarting the llama-server each time, and record for each
value:
- peak VRAM from `nvidia-smi --query-gpu=memory.used --format=csv -l 1` during a
  request;
- decode tokens/s and prompt tokens/s from the llama-server timings;
- host RAM in use (experts on CPU live in system RAM).
Keep at least 512 MiB of VRAM headroom, the margin the `scripts/model_bench`
fit clause uses. Pick the lowest value that fits, write it into `.env`, and
record the sweep in the PR or the vault note. The number of expert layers in
gpt-oss-20b is UNVERIFIED in this repo; the llama-server load log states it.

6. Tailscale exposure
---------------------

The host profile publishes no ports. The service shares the sidecar's network
namespace, so it is reachable only on the tailnet, at the sidecar's hostname,
port 8090. The llama-server is reachable only inside the compose network.

- Create a tagged, non-ephemeral auth key in the tailnet admin console and put
  it in `TS_AUTHKEY`. Use an ACL that allows ONLY the main MIST machine to reach
  this host on tcp/8090. [UNVERIFIED: ACL syntax against your tailnet policy]
- Start the rest of the profile:

      docker compose -f docker-compose.extraction.yml --profile extraction-host up -d

- From the MIST machine, check the contract endpoints:

      # TS_HOSTNAME defaults to mist-extraction-gtx1070 in the compose file
      curl -s http://<TS_HOSTNAME>:8090/v1/health
      curl -s http://<TS_HOSTNAME>:8090/v1/info

  `/v1/info` must report `extraction_version`, `model_hash` (your
  `EXTRACTION_MODEL_HASH`), `llama_cpp_build` `b11151` and `location_label`
  `host-1070`.
- On the MIST machine, set `MIST_EXTRACTION_SERVICE_URL=http://<TS_HOSTNAME>:8090`
  for the backend and restart it. The main backend container must itself be on
  the tailnet (host networking through the host's Tailscale client, or a sidecar
  of its own). [UNVERIFIED: how the main backend container reaches the tailnet
  today; decide and record it]

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
