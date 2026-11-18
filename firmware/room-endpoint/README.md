<!-- SPDX-License-Identifier: MIT -->
<!-- Part of MIST.AI, released under the MIT License; see LICENSE at the repository root. -->

# MIST room endpoint firmware

MIST's own firmware for a Voice PE: ESPHome on the ESP32-S3 with MIST's components
(`components/`), one persistent `wss://` link to MIST's device listener, and nothing personal in
this directory. A device's Wi-Fi details, token, CA certificate and name model live in its private
directory, `<private dir>/<device>/`, outside every work tree; the built images are copied there.

## H4: build the spike, flash it, and measure the device gate

The spike build measures the device gate G1 against the spike listener, a stand-in for MIST's
backend on your PC. Its wake is the stock "stop" model (approved for H4 only): a "stop" said with
nothing playing opens the uplink. It reports `device_stats` every 10 s, plays the listener's tone
through a minimal spike-only player (no fades or ducking; the room build's player replaces it),
and keeps an uplink open up to 75 s, so 60 s of playback fits in one wake. It needs H3 done: the devices dir, the firewall
rule, and `devices.py add <device>` with your Wi-Fi lines added to its `secrets.yaml`.

Placeholders: `<repo>` is a mist.ai checkout, `<private dir>/<device>` the device's private
directory, `<devices dir>` the devices dir from `devices.py init`, `<result dir>` a directory for
the result, `<PC LAN IP>` the PC's reserved address, and `<worker image>` the mist.ai worker image
(ESPHome, ESP-IDF and the firmware assets are in it, so the build needs no network).

1. Make the spike YAML. Copy `personal.example.yaml` to `<private dir>/<device>/spike.yaml`, set
   `device_name: <device>`, change the include to `../room-endpoint/spike.yaml`, and delete the
   `wake_model` line.

2. Build the spike image. The repository is mounted read-only; the build runs in a throwaway
   directory under the container's `/tmp` and copies the images into the private directory:

   ```bash
   docker run --rm --network none -v <repo>:/repo:ro -v <private dir>/<device>:/private/<device> -w /repo <worker image> python -m scripts.room_endpoint.build compile spike --config /private/<device>/spike.yaml --out /private/<device>/images/spike
   ```

   It prints the two image paths: `firmware.factory.bin` and `firmware.ota.bin`, in
   `<private dir>/<device>/images/spike/`. A compile takes about 3 minutes.

3. Flash the factory image. Connect the Voice PE to the PC by USB, open https://web.esphome.io in
   Chrome or Edge, choose Connect, pick the device's serial port, choose Install, and select
   `<private dir>/<device>/images/spike/firmware.factory.bin`. Later builds can go over the air
   with the OTA password in `secrets.yaml`.

4. Start the spike listener on the device port. `-i` keeps stdin open for the operator commands:

   ```bash
   docker run --rm -i -p <PC LAN IP>:8443:8443 -v <repo>:/repo:ro -v <devices dir>:/devices:ro -v <result dir>:/result -w /repo <worker image> python -m scripts.room_endpoint.spike_listener --devices /devices --port 8443 --out /result/spike_result.json
   ```

   The device connects within 15 s of booting. Type one command per line; each prints a one-line
   reply:

   - `state <idle|attend|listen|think|speak>` sends `state_cycle`; `speak` and `idle` close the
     device's uplink.
   - `reject [reason]` sends `voice_decision` `wake_rejected` (default `not_addressed`), which
     closes the uplink.
   - `tone` streams a 400 Hz tone for 60 s to the device.
   - `status` prints the counts so far.
   - `quit` writes the result and stops (so does `docker stop`).

5. Measure, in this order:

   - Wakes: say "stop" 20 times at 1 to 3 m, with nothing playing. After each, wait a few seconds
     and type `state idle` to close the uplink.
   - Playback: say "stop" with nothing playing (it opens the uplink), then type `tone` at once.
     Do not say "stop" during the tone: during playback "stop" is a `barge_in`, not a wake. Let the
     60 s play out, then type `state idle`.
   - Soak: leave it running for 30 minutes in all.
   - Reconnection: `quit`, then start the listener again with the same command and the same
     `--out`, and `quit` once the device has reconnected.

   The result is `<result dir>/spike_result.json`: counts and timings only, never audio. G1's
   checks and a pass or fail for each are in it.

## The room build

`room.yaml` wakes on the name model alone (`wake_model`), which the private YAML sets; nothing else
differs in the hardware. Copy `personal.example.yaml` to `<private dir>/<device>/room.yaml`, set
`device_name` and `wake_model`, and build with the models directory mounted at `/models`:

```bash
docker run --rm --network none -v <repo>:/repo:ro -v <private dir>/<device>:/private/<device> -v <models dir>:/models:ro -w /repo <worker image> python -m scripts.room_endpoint.build compile room --config /private/<device>/room.yaml --out /private/<device>/images/room
```

## Layout and how a build works

- `room.yaml` and `spike.yaml`: the two top-level configs, with every substitution's default set
  to the worker image's path (`room.yaml` alone has no default name model).
- `packages/device.yaml`: the hardware, adapted from the published Voice PE config (MIT): board,
  PSRAM, XMOS front end (`voice_kit`, referenced at its pinned commit, never copied), I2S buses,
  microphone, speaker chain, Wi-Fi, OTA, SNTP and the wake models.
- `packages/link.yaml`: `mist_link`, the link to MIST; `packages/spike.yaml` adds the spike's
  `device_stats`, and `room.yaml` never includes it.
- `components/mist_core/`: the portable core, tested on the host (`tests/unit/firmware`);
  `components/mist_link/`: the link, through `esp_websocket_client` added from the image's local
  copy.

`scripts/room_endpoint/build.py` copies this directory to a new directory under `/tmp`, sets
ESPHome's data directory and `HOME` inside it, serves the image's assets on `127.0.0.1:8765` for
the compile (`voice_kit` takes the XMOS image only by URL), runs `esphome compile`, and copies the
images out with `--out`. With `--config`, the private YAML's directory is copied beside this one,
so its `!include ../room-endpoint/...` resolves and its own `secrets.yaml` and `mist_ca.pem` are
read. Without `--config` it builds the repository's config with throwaway secrets, which is the
compile check a change here runs:

```bash
python -m scripts.room_endpoint.build compile spike
```
