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
   `<private dir>/<device>/images/spike/firmware.factory.bin`. Later builds go over the air
   instead ("Over the air: later builds", below).

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
     The listener replies `tone started (audio_tag <n>, 60 s)` and the speaker plays a steady
     400 Hz tone. Do not say "stop" during the tone: during playback "stop" is a `barge_in`, not a
     wake. Until the tone ends, type no `state` or `reject` and leave the mute switch alone: each
     closes the uplink, and G1 counts 60 s of uplink beside the tone. The listener prints nothing
     when the tone ends, so type `status`: it shows `tone=playing` until the tone's last frame
     has gone out, then `tone=idle`. Once it shows `tone=idle`, type `state idle`.
   - Soak: leave it running for 30 minutes in all.
   - Reconnection: `quit`, then start the listener again with the same command and the same
     `--out`, and `quit` once the device has reconnected.

   The result is `<result dir>/spike_result.json`: counts and timings only, never audio. G1's
   checks and a pass or fail for each are in it.

6. The device's log, if a check fails. It comes over USB only (the firmware has no native API):
   connect the Voice PE by USB, open https://web.esphome.io, choose Connect, pick the serial port
   and choose Logs. Each uplink logs one line when it opens and one when it closes:

   - `Uplink open: wake '<word>', backlog <ms> ms, first seq <n>`;
   - `Uplink closed: <cause>, next seq <n>`, where `<cause>` is `state_cycle speak`,
     `state_cycle idle`, `wake_rejected`, `wake_lost`, `dropped`, `cap`, `mute` or
     `disconnection`. For an operator command it reads as the result's `closed_by`, so a close
     the listener did not cause (`mute`, `cap`, `disconnection`) shows here and not there.

## Over the air: later builds

After the first USB flash, a new image goes to the device over Wi-Fi with ESPHome's own OTA,
authenticated by `mist_ota_password` from the device's `secrets.yaml`, which the private YAML
reads through `packages/device.yaml` (`ota: password: !secret mist_ota_password`).

1. Build the new image with step 2's command (or the room build's). The OTA image is
   `<private dir>/<device>/images/spike/firmware.ota.bin` (`images/room/` for the room build).

2. Find the device's IP address on Windows, in any of these ways:

   - In PowerShell, `ping -4 <device>.local`: the firmware answers multicast DNS under its
     `device_name`, and the reply line shows the address, if Windows resolves `.local` names on
     your network.
   - Your router's list of DHCP clients: the device appears under `<device>`. Reserving that
     address there keeps it the same after a reboot.
   - The device's log over USB (step 6): the Wi-Fi component logs `IP Address:` once it connects.

3. Upload. This runs in the worker image with the network on, so the container reaches the
   device; it mounts the repository's config and the private directory read-only where `build.py`
   puts them, passes the one substitution `build.py` passes for a `--config` build
   (`mist_components_path`), and serves the image's assets on `127.0.0.1:8765` as `build.py`
   does, because ESPHome validates the whole config, `voice_kit`'s XMOS image by URL included,
   before it uploads:

   ```bash
   docker run --rm -v <repo>/firmware/room-endpoint:/build/room-endpoint:ro -v <private dir>/<device>:/build/config:ro -e HOME=/tmp -e ESPHOME_DATA_DIR=/tmp/esphome -w /tmp <worker image> sh -c "/opt/python312/bin/python3.12 -I -m http.server 8765 --bind 127.0.0.1 --directory /opt/mist-firmware-deps >/dev/null 2>&1 & sleep 2; esphome -s mist_components_path /build/room-endpoint/components upload /build/config/spike.yaml --device <device IP> --file /build/config/images/spike/firmware.ota.bin"
   ```

   For the room build, add `-v <models dir>:/models:ro` and use `room.yaml` and
   `images/room/firmware.ota.bin`. ESPHome prints `Successfully uploaded program.` and the device
   restarts into the new image, then connects to the listener again.

## The room build

`room.yaml` wakes on the name model alone (`wake_model`), which the private YAML sets; nothing else
differs in the hardware. Copy `personal.example.yaml` to `<private dir>/<device>/room.yaml`, set
`device_name` and `wake_model`, and build with the models directory mounted at `/models`:

```bash
docker run --rm --network none -v <repo>:/repo:ro -v <private dir>/<device>:/private/<device> -v <models dir>:/models:ro -w /repo <worker image> python -m scripts.room_endpoint.build compile room --config /private/<device>/room.yaml --out /private/<device>/images/room
```

## The pre-wake ring and wake uplink

`mist_prewake` (configured in `packages/link.yaml`) holds the last `ring_ms` (6 s) of microphone
channel 0, the XMOS chain through AGC at gain 1 (as the published config's voice pipeline takes
it, `home-assistant-voice.yaml` lines 1814-1820; the wake models read channel 1 at gain 4, lines
1721-1726), in a PSRAM ring (`heap_caps_malloc` with
`MALLOC_CAP_SPIRAM`, 16 kHz mono PCM16 in 20 ms frames, through `mist_core::PcmRing`). Nothing
leaves the device before a name detection. A detection by the name model, while connected,
unmuted and with `voice_state.active` true, sends `wake {keyword: "mist", score, backlog_ms}`,
then the newest `backlog_ms` (at most 5000) of the ring oldest first, then live frames, all
through `mist_link`. `mist_core::Uplink` (`components/mist_core/uplink.cpp`) is the state machine,
tested on the host in `tests/unit/firmware/cpp/test_uplink.cpp`.

A wake queues the whole backlog at once (250 frames, about 175 KB on the wire once WebSocket and
TLS framing are added), and G1 wants its last frame at the server within 1 s. Two settings carry
it. `mist_prewake` requests ESPHome's high-performance networking, as the published config's
Sendspin does: without it lwIP's send buffer is ESP-IDF's 5760 bytes, so about 30 round trips
carry one backlog. And `packages/device.yaml` turns Wi-Fi power save off: in ESPHome's default
modem sleep the server's ACKs can wait for the next beacon, which stalls the send window.

Which detection opens it. `micro_wake_word`'s `on_wake_word_detected` passes only the detected
model's word, so `mist_prewake` opens the uplink only when that word is the name model's
(`wake_model: mist_wake`, the model `${wake_model}` loads). Its codegen refuses any build in which
another model carries the same word, except the spike build, whose stand-in wake is the stop
model's own file; it recognises the spike build by `mist_link: device_stats`, which only
`packages/spike.yaml` sets. So in the room build only `${wake_model}` opens the uplink and "stop"
never does. This is checked at build time (`esphome config` refuses such a build), not by a host
test. A room compile without `--config` cannot take the stop manifest as `wake_model`, so
`build.py` writes a stand-in name model into its throwaway build directory: the image's stop
manifest with its word changed to "compile check" and its `model` naming the image's stop model
file. That image still opens on "stop", so `build.py` refuses `--out` for it; a room image leaves
the container only when built from the private YAML with `--config`.

The uplink closes on the first of these, checked against the realtime controller on `main`:

- `state_cycle` `speak` or `idle` after the wake: the controller sends `state_cycle speak` before
  a reply stretch's first `0x01` (`backend/realtime/turn.py:1278-1285`) and `state_cycle idle`
  at a turn's end (`turn.py:2972`) or after an utterance that opened no turn (`turn.py:3067`);
- `voice_decision` `dropped` (`backend/realtime/protocol.py:849-874`, which closes the utterance
  it names, reason `muted` included), `wake_lost` (`protocol.py:922-932`, the same) or
  `wake_rejected` (`protocol.py:935-948`);
- `uplink_cap_ms` (30 s; 75 s in the spike build) after the wake;
- the mute switch;
- disconnection.

It stays open through `voice_decision` `endpoint`, which opens a turn while `hold` and
`supersede` still listen (`protocol.py:789-826`), and through `state_cycle` `think`, so the
server's hold and supersede logic still hears a sentence that goes on.

The mute switch is GPIO3, from the published Voice PE config this firmware adapts
(`home-assistant-voice.yaml` at 2644f4c, line 523, the `gpio` binary sensor
`hardware_mute_switch`; on means muted). Turning it on clears the ring, closes any uplink and
sends `mute {active: true}`; turning it off clears the ring again and sends `mute {active:
false}`, so no audio from before or during a mute reaches a later backlog. A `mute` the link's
queue refuses is sent again on the next pass, and always before the next `wake`; each
`voice_start` carries the current state.

`wake.score` is the name model's probability cutoff, not the detection's probability. ESPHome
2026.10.2 computes a detection's probabilities (`DetectionEvent.max_probability` and
`average_probability`, `esphome/components/micro_wake_word/streaming_model.h:20-28`) but only
logs them; its trigger passes the word alone
(`this->wake_word_detected_trigger_.trigger(*detection_event.wake_word)`, `micro_wake_word.cpp:496`;
the automation's one argument is `wake_word`, `micro_wake_word/__init__.py:600-605`), and the
detection queue is protected (`micro_wake_word.h:138`). So `mist_prewake` sends the cutoff the model runs with,
`WakeWordModel::get_probability_cutoff() / 255`: the manifest's `micro.probability_cutoff`, or the
model's `probability_cutoff` in the YAML (`micro_wake_word/__init__.py:555-558`), quantized to
1/255. The server's calibration then ranks the device at its cutoff's false-accept rate. H4's
spike build shows it: the stop manifest's cutoff is 0.5, quantized to `int(0.5 * 255)` = 127, and
every wake carried `score` 127/255 = 0.49803922.

## Layout and how a build works

- `room.yaml` and `spike.yaml`: the two top-level configs, with every substitution's default set
  to the worker image's path (`room.yaml` alone has no default name model).
- `packages/device.yaml`: the hardware, adapted from the published Voice PE config (MIT): board,
  PSRAM, XMOS front end (`voice_kit`, referenced at its pinned commit, never copied), I2S buses,
  microphone, speaker chain, mute switch, Wi-Fi, OTA, SNTP and the wake models.
- `packages/link.yaml`: `mist_link`, the link to MIST, and `mist_prewake`, the pre-wake ring and
  wake uplink; `packages/spike.yaml` adds the spike's `device_stats`, and `room.yaml` never
  includes it.
- `components/mist_core/`: the portable core, tested on the host (`tests/unit/firmware`);
  `components/mist_link/`: the link, through `esp_websocket_client` added from the image's local
  copy; `components/mist_prewake/`: the microphone tap, wake hook and mute switch around
  `mist_core`'s ring and uplink.

`scripts/room_endpoint/build.py` copies this directory to a new directory under `/tmp`, sets
ESPHome's data directory and `HOME` inside it, serves the image's assets on `127.0.0.1:8765` for
the compile (`voice_kit` takes the XMOS image only by URL), runs `esphome compile`, and copies the
images out with `--out`. With `--config`, the private YAML's directory is copied beside this one,
so its `!include ../room-endpoint/...` resolves and its own `secrets.yaml` and `mist_ca.pem` are
read. Without `--config` it builds the repository's config with throwaway secrets and, for
`room`, the stand-in name model above (no `--out`), which is the compile check a change here
runs:

```bash
python -m scripts.room_endpoint.build compile spike
python -m scripts.room_endpoint.build compile room
```
