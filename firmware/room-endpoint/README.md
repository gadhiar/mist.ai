<!-- SPDX-License-Identifier: MIT -->
<!-- Part of MIST.AI, released under the MIT License; see LICENSE at the repository root. -->

# MIST room endpoint firmware

MIST's own firmware for a Voice PE: ESPHome on the ESP32-S3 with MIST's components
(`components/`), one persistent `wss://` link to MIST's device listener, and nothing personal in
this directory. A device's Wi-Fi details, token, CA certificate and name model live in its private
directory, `<private dir>/<device>/`, outside every work tree; the built images are copied there.

Placeholders: `<repo>` is a mist.ai checkout, `<private dir>/<device>` the device's private
directory, `<device>` the device's registry name, `<devices dir>` the devices dir from
`devices.py init`, `<result dir>` a directory for the result, `<PC LAN IP>` the PC's reserved
address, `<device IP>` the device's address, `<models dir>` the directory that holds the name
model's `<model id>/device/` directory, and `<worker image>` the mist.ai worker image (ESPHome,
ESP-IDF and the firmware assets are in it, so a build needs no network).

The order for a new device: the private directory, then the H4 spike (once, to measure the device
gate), then the room build. A device that already runs the spike image goes to the room image over
the air ("The room build", step 3).

## The private directory

Each device has one directory, outside every git work tree, which only your own commands read:

```
<private dir>/<device>/
├── secrets.yaml        # devices.py add: token, server address and port, OTA password; plus Wi-Fi
├── mist_ca.pem         # devices.py add: the CA the device checks the listener's certificate by
├── spike.yaml          # the H4 spike build's YAML, from personal.example.yaml
├── room.yaml           # the room build's YAML, from personal.example.yaml
└── images/
    ├── spike/          # build.py --out: firmware.factory.bin, firmware.ota.bin
    └── room/           # build.py --out: firmware.factory.bin, firmware.ota.bin
```

`build.py --config` copies this directory, without `images/`, into its throwaway build directory
beside a copy of `firmware/room-endpoint/`, so the YAML's `!include ../room-endpoint/...` resolves
and its own `secrets.yaml` and `mist_ca.pem` are read (`scripts/room_endpoint/build.py`, module
docstring).

To make it, on the PC:

1. Reserve an address for the PC and one for the device in the router's DHCP settings: the
   server certificate names the PC's address, and a changed address needs `devices.py cert`.

2. Allow the device port through Windows Firewall, private profile only, from the LAN. In an
   administrator PowerShell:

   ```powershell
   New-NetFirewallRule -DisplayName "MIST device listener" -Direction Inbound -Protocol TCP -LocalPort 8443 -Profile Private -RemoteAddress LocalSubnet -Action Allow
   ```

   `Remove-NetFirewallRule -DisplayName "MIST device listener"` takes it away again.

3. Make the devices dir and the CA once with `devices.py init`, as `deploy/daily/README.md`
   ("Room devices", "Provisioning") gives it; that section also names the devices dir and the CA
   dir and where each lives.

4. Register the device and write its private files:

   ```bash
   python scripts/daily/devices.py add <device> --area <area> --out <private dir>/<device>
   ```

   `<device>` is a lowercase letter then up to 23 lowercase letters, digits or hyphens, and not
   `desktop`, `all` or `room`; `<area>` is the area its lights commands act on
   (`scripts/daily/devices.py`, `NAME_PATTERN`, `RESERVED_NAMES`). It writes `secrets.yaml`
   (`mist_device_id`, `mist_device_token`, `mist_server_host`, `mist_server_port`,
   `mist_ota_password`) and `mist_ca.pem`, and never prints the token.

5. Add the Wi-Fi lines to that `secrets.yaml`, below the ones `add` wrote; `add` and `rotate` keep
   them:

   ```yaml
   wifi_ssid: "<Wi-Fi name>"
   wifi_password: "<Wi-Fi password>"
   ```

## H4: build the spike, flash it, and measure the device gate

The spike build measures the device gate G1 against the spike listener, a stand-in for MIST's
backend on your PC. Its wake is the stock "stop" model (approved for H4 only): a "stop" said with
nothing playing opens the uplink. It reports `device_stats` every 10 s, plays the listener's tone
through a minimal spike-only player (no fades or ducking; the room build's player replaces it),
and keeps an uplink open up to 75 s, so 60 s of playback fits in one wake. It needs the private
directory above.

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

## The room build

`room.yaml` wakes on the name model alone (`wake_model`), which the private YAML sets, and adds
MIST's player, controls and alarms (`packages/player.yaml`, `packages/controls.yaml`,
`packages/alarm.yaml`); the hardware is the same as the spike build's. The device connects to
MIST's own device listener, so the backend runs with it on first: `deploy/daily/README.md`, "Room
devices", gives the `daily.env` settings that turn it on.

1. Make the room YAML. Copy `personal.example.yaml` to `<private dir>/<device>/room.yaml` and set
   two lines: `device_name: <device>`, and `wake_model` to the name model's manifest as the
   container sees it, `/models/<model id>/device/<model file>.json`, with its `.tflite` beside
   it. Keep the include, `../room-endpoint/room.yaml`.

2. Build the room image, with `<models dir>` mounted read-only at `/models`:

   ```bash
   docker run --rm --network none -v <repo>:/repo:ro -v <private dir>/<device>:/private/<device> -v <models dir>:/models:ro -w /repo <worker image> python -m scripts.room_endpoint.build compile room --config /private/<device>/room.yaml --out /private/<device>/images/room
   ```

   It prints `factory image:` and `ota image:` with their paths in
   `<private dir>/<device>/images/room/`. `build.py` refuses `--out` for a room build without
   `--config`, so a room image leaves the container only when built from the private YAML.

3. Put it on the device, one of two ways:

   - The device runs the spike image (after H4): update it over Wi-Fi with the room image's
     upload command in "Over the air: later builds", below. The spike and room images read the
     same `secrets.yaml`, so the OTA password is the same.
   - A device with no MIST image yet: flash
     `<private dir>/<device>/images/room/firmware.factory.bin` over USB through
     https://web.esphome.io, as H4 step 3 does.

4. Check it. The device restarts into the room image and connects to the listener. While the
   link is down the LED ring twinkles red; once connected and idle it is off ("Using the device",
   below). On the USB log (H4 step 6), `mist_link` logs `Connected` when the link comes up, and
   `Link error: handshake status <n>, TLS error <code>, socket errno <n>` when an attempt fails.

## Over the air: later builds

After the first USB flash, a new image goes to the device over Wi-Fi with ESPHome's own OTA,
authenticated by `mist_ota_password` from the device's `secrets.yaml`, which the private YAML
reads through `packages/device.yaml` (`ota: password: !secret mist_ota_password`).

1. Build the new image with its build command (H4 step 2, or "The room build" step 2). The OTA
   image is `<private dir>/<device>/images/spike/firmware.ota.bin`, or `images/room/` for the
   room build.

2. Find the device's IP address on Windows, in any of these ways:

   - In PowerShell, `ping -4 <device>.local`: the firmware answers multicast DNS under its
     `device_name`, and the reply line shows the address, if Windows resolves `.local` names on
     your network.
   - Your router's list of DHCP clients: the device appears under `<device>`. Reserving that
     address there keeps it the same after a reboot.
   - The device's log over USB (H4 step 6): the Wi-Fi component logs `IP Address:` once it
     connects.

3. Upload. This runs in the worker image with the network on, so the container reaches the
   device; it mounts the repository's config and the private directory read-only where `build.py`
   puts them, passes the one substitution `build.py` passes for a `--config` build
   (`mist_components_path`), and serves the image's assets on `127.0.0.1:8765` as `build.py`
   does, because ESPHome validates the whole config, `voice_kit`'s XMOS image by URL included,
   before it uploads. For the spike image:

   ```bash
   docker run --rm -v <repo>/firmware/room-endpoint:/build/room-endpoint:ro -v <private dir>/<device>:/build/config:ro -e HOME=/tmp -e ESPHOME_DATA_DIR=/tmp/esphome -w /tmp <worker image> sh -c "/opt/python312/bin/python3.12 -I -m http.server 8765 --bind 127.0.0.1 --directory /opt/mist-firmware-deps >/dev/null 2>&1 & sleep 2; esphome -s mist_components_path /build/room-endpoint/components upload /build/config/spike.yaml --device <device IP> --file /build/config/images/spike/firmware.ota.bin"
   ```

   For the room image, which also reads the name model while it validates:

   ```bash
   docker run --rm -v <repo>/firmware/room-endpoint:/build/room-endpoint:ro -v <private dir>/<device>:/build/config:ro -v <models dir>:/models:ro -e HOME=/tmp -e ESPHOME_DATA_DIR=/tmp/esphome -w /tmp <worker image> sh -c "/opt/python312/bin/python3.12 -I -m http.server 8765 --bind 127.0.0.1 --directory /opt/mist-firmware-deps >/dev/null 2>&1 & sleep 2; esphome -s mist_components_path /build/room-endpoint/components upload /build/config/room.yaml --device <device IP> --file /build/config/images/room/firmware.ota.bin"
   ```

   ESPHome prints `Successfully uploaded program.` and the device restarts into the new image,
   then connects to the listener again.

## Using the device

### The LED ring

The player picks one pattern from MIST's state (`mist_core::UiState::led()`,
`components/mist_core/ui_state.cpp`); a lower number wins over a higher one. Rows 4 to 8 follow
her `state_cycle`, one at a time, and a lost link forgets it, so a new session starts idle. Each
pattern is an effect of `mist_leds` (`packages/controls.yaml`), drawn in the colour and brightness
`components/mist_player/mist_player.cpp` (`look_for`) gives it: red for mute and a lost link, the
published Voice PE config's light blue (RGB 0.094, 0.733, 0.949) for the rest, at 66% brightness
except a ring at 100%.

| Priority | Pattern | The ring | It means |
|---|---|---|---|
| 1 | `Ringing` | the whole ring pulses, light blue, full brightness | an alarm rings |
| 2 | `Muted` | the two lights beside the microphones, steady red | the mute switch is on |
| 3 | `Disconnected` | red lights twinkle around the ring | the link to MIST is down |
| 4 | `Attend` | two light blue arms turn slowly clockwise | she heard her name |
| 5 | `Listen` | the same turn, twice as fast | she is listening |
| 6 | `Think` | two opposite lights pulse, light blue | she is thinking |
| 7 | `Speak` | two light blue arms turn anticlockwise | she is speaking |
| 8 | `Idle` | off | connected and idle |

### The controls

Wired in `packages/controls.yaml`; what each does is `UiState` (`components/mist_core/ui_state.h`
and `ui_state.cpp`).

- The center button. No press opens the microphone: there is no push-to-talk.
  - Single press (held under 1 s): during a ring, stops it; while she speaks, sends MIST
    `interrupt`, which stops the current turn; otherwise nothing.
  - Double press (two presses under 1 s each, at most 0.25 s apart): during a ring, snoozes it;
    otherwise nothing.
  - Long press (held 1 s or more): nothing.
- The dial sets the volume, from 0 to 1, by 0.05 per detent: clockwise up, anticlockwise down,
  held at 0 and 1. At 0 the speaker is silent. With nothing saved it starts at 0.724, the loudest
  setting at which no reply clips in the DAC (0 dB). Each change is saved to flash and restored at
  boot. The DAC level is `0.4 + 0.45 * volume`, -28.5 dB to +10.5 dB, the published Voice PE
  config's range. A ring plays at the dial's volume or at least 0.6 of full volume, whichever is
  higher (`ring_volume_floor` in `room.yaml`), and never above 0 dB.
- "Stop". The stop model listens only while the mute switch is off and either a ring plays or she
  speaks. Said while she speaks, it sends MIST `barge_in`, and MIST's barge-in rules
  (`backend/realtime/bargein.py`) decide whether she stops; said during a ring, it stops the
  ring. At other times nothing hears it.
- The mute switch (on the side). On: the name and stop models are off and no audio leaves the
  device, so on the device a ring can then be stopped only with the button. Its effect on the
  uplink is in "The pre-wake ring and wake uplink", below.

### Alarms on the device

`components/mist_alarm/` and `components/mist_core/alarm_rules.h`, configured in
`packages/alarm.yaml`. An alarm rings on this device when its `ring_on` is `all` or names
`<device>` (`mist_device_id`, the registry name, from `secrets.yaml`).

- The ring: four 100 ms beeps of 880 Hz every 1.4 s, synthesised on the device, with the LED ring
  pulsing. A single press or "stop" stops it; a double press snoozes it.
- With the link up, MIST's `alarm_fired` rings at once, and the device reports that it rang. Stop
  and snooze go to MIST as `alarm_action`, and MIST applies them everywhere. A ring stopped
  elsewhere stops here too, and a ring ends by itself after the alarm's `ring_timeout_s`.
- With the link down, the device rings from its copy of the schedule, MIST's last `alarm_state`
  (at most the 16 earliest of this device's alarms), kept in flash so a reboot or a power cut
  keeps it. It rings an occurrence when the link was down for the whole span from its fire time
  to `fallback_after_s` after it, and only until `missed_grace_s` after the fire time. No
  occurrence rings twice on this device, and it does not ring a snoozed local ring again.
- Reports: on reconnection the device sends each local ring's outcome to MIST (`rang`, `stopped`,
  or `snoozed` with the time it was snoozed until), and sends it again on each later connection
  until MIST answers it. Outcomes are kept in flash too.
- The clock: SNTP once it has synced, else MIST's time from the last `alarm_state` or
  `alarm_fired`. Nothing rings locally until one of them has set it, so a device that restarts
  while it can reach neither an SNTP server nor MIST does not ring.

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

## Pins

Every external part of an image is fixed in the worker image, under `/opt`. Each pin below was
read there: `/opt/mist-firmware-deps/PINS.txt` records the versions and the sha256 of each
downloaded file, and the other files named hold the rest. To read them from your copy of the image:

```bash
docker run --rm --network none <worker image> cat /opt/mist-firmware-deps/PINS.txt
```

- ESPHome 2026.10.2: `PINS.txt` and `esphome version`; `packages/device.yaml` sets it as
  `min_version`.
- ESP-IDF 5.5.6: `/opt/esp-idf` links to `/opt/espressif/frameworks/5.5.6`, whose `version.txt`
  reads `v5.5.6`.
- The published Voice PE config, `esphome/home-assistant-voice-pe` at
  `2644f4c794271d735d182ca7ebf899ed46164f4e` (`PINS.txt`), checked out at
  `/opt/mist-firmware-deps/home-assistant-voice-pe/`. `packages/*.yaml` adapt its
  `home-assistant-voice.yaml`.
- `voice_kit`, the XMOS front end's component: that checkout's `esphome/components/voice_kit`,
  which `voice_kit_path` names; referenced there, never copied into this repository.
- `esp_websocket_client` 1.8.0 (`PINS.txt`), from `espressif/esp-protocols` at
  `70bf122fc8bc74622dcf7e15233b5b2af9088df5` (its `idf_component.yml`), at
  `/opt/mist-firmware-deps/esp_websocket_client`, which `websocket_client_path` names.
- The XMOS firmware, ffva v1.3.1: `/opt/mist-firmware-deps/xmos/ffva_v1.3.1_upgrade.bin`, sha256
  `11b563d6c51424f49f1e1d3585ba03a073e2802798e6aa067c6734d7ecb6935b` (`PINS.txt`); the
  published config fetches the same release from `esphome/voice-kit-xmos-firmware` (line 1702).
  `voice_kit` takes it only by URL, so `build.py` serves it on `127.0.0.1:8765`.
- The `stop` model: `kahrendt/microWakeWord` release `stop` (`PINS.txt`; the published config's
  line 1735), at `/opt/mist-firmware-deps/models/stop/`: `stop.json` sha256
  `bd13aeb1b83852649dc4fb6135cb160ff68716d14612b06f6a405342c57447aa`, `stop.tflite` sha256
  `b5a18c4ad681a89950dfade31011e1631bdcb333e93c84519a1a63ff4f071146`; cutoff 0.5, tensor arena
  21,000 bytes.
- The VAD model: `esphome/micro-wake-word-models` at
  `9a72109cc935f77e34cf8ffe7b868155c37ec201` (`PINS.txt`), at
  `/opt/mist-firmware-deps/models/vad/`: `vad.json` sha256
  `bd8c9cd4350814f761e651685ae5196e2a56521aaf5670071ba742927de58ed7`, `vad.tflite` sha256
  `7aa4db6d5fb7c5358609f6931e7847d303c16a43d178638bc14104f50d7eff5f`; tensor arena 16,772 bytes.

Their licences are in `NOTICE` at the repository root.

## Layout and how a build works

- `room.yaml` and `spike.yaml`: the two top-level configs, with every substitution's default set
  to the worker image's path (`room.yaml` alone has no default name model). `room.yaml` includes
  the device, link, player, controls and alarm packages; `spike.yaml` the device, link and spike
  packages.
- `personal.example.yaml`: the template for a device's private YAML.
- `packages/device.yaml`: the hardware, adapted from the published Voice PE config (MIT): board,
  PSRAM, XMOS front end (`voice_kit`, referenced at its pinned commit, never copied), I2S buses,
  microphone, speaker chain, mute switch, Wi-Fi, OTA, SNTP and the wake models.
- `packages/link.yaml`: `mist_link`, the link to MIST, and `mist_prewake`, the pre-wake ring and
  wake uplink; `packages/spike.yaml` adds the spike's `device_stats` and its spike-only player,
  and `room.yaml` never includes it.
- `packages/player.yaml`: `mist_player`, which plays MIST's replies through the reply resampler,
  the mixer and the DAC, owns the volume, and drives the models and the LED ring from MIST's state.
- `packages/controls.yaml`: the button, the dial, the stop model's detections and the LED ring's
  hardware and effects, wired to the player and the alarm component.
- `packages/alarm.yaml`: `mist_alarm`, MIST's alarms on the device.
- `components/mist_core/`: the portable core, plain C++ tested on the host
  (`tests/unit/firmware`): the protocol codec (`codec`), a JSON reader and writer (`json_lite`),
  the pre-wake ring (`ring.h`), the uplink state machine (`uplink`), the playback tag rules
  (`tag_queue`), the LEDs, controls and volume (`ui_state`), and the alarm rules, clock and tone
  (`alarm_rules`).
- `components/mist_link/`: the link, through `esp_websocket_client` added from the image's local
  copy; `components/mist_prewake/`: the microphone tap, wake hook and mute switch around
  `mist_core`'s ring and uplink; `components/mist_player/`: the reply player, level, models and
  LEDs around `mist_core`'s tag queue and `UiState`; `components/mist_alarm/`: the alarm glue
  between the link, `mist_core`'s alarm rules, the alarm speaker and the player.

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
