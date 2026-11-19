# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""Host tests for the room endpoint's configs, its link component and `build.py`.

The device compile itself runs as its own step (`python -m scripts.room_endpoint.build compile
spike`), because it takes minutes; these tests pin what can be checked on the host: the name scan
over every firmware file, the shape of the YAML packages, the spike-only parts, and the argument
lists, environment and file copies `build.py` makes, with a fake compile.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from scripts.room_endpoint import build

REPO_ROOT = Path(__file__).resolve().parents[3]
FIRMWARE_ROOT = REPO_ROOT / "firmware"
ENDPOINT = FIRMWARE_ROOT / "room-endpoint"
PACKAGES = ENDPOINT / "packages"
LINK_DIR = ENDPOINT / "components" / "mist_link"
README = ENDPOINT / "README.md"

# The sha256 of each stock wake model id in the pinned Voice PE config and of each name in those
# ids, lowercased. Only digests are kept, so no such name enters the repository.
FORBIDDEN_DIGESTS = frozenset(
    {
        "1d420a08d4f65805513a94660f01325357236776cf41bec9a380ca147518db66",
        "2bb5ae85816deab84c206210b2b533b676048f1d324802fea347ae3741e2e08a",
        "3784b6419c4847326e42879630ce0a79f3706916c6f837b205f9181b66cb187c",
        "7d9c6fc5dade15956ae6d39b035b62d9e47067bf2adf4d6574edc096ac0e2acb",
        "9a7966105906d8ec7b0a75b1544c44fa399efc9c919904f028c3d2a297a48894",
        "cb5ef041a5a2dcf0ab092d52085f95cb077338a3ab26abb73a455d778ce39ff8",
        "d2b4ea1d24cfbd1d1fe9ec4094ef967a35edf0076e374255a7ef587a6f0e29ae",
        "d45781b68a5bed85fa011152729a4912fa06409aa2e204970525bc70a1f6278b",
        "dda820158d142a0b73380cd3efa9ff7d9fa11336e9ba4b3693098964537d61a3",
        "e43539a8857ac00620f4746226b04b33d388b97b6c1fa4b824e9e20f9fe25620",
        "e63bd702d0a32236392c0983c2c276edc30ff68988fac8bb5b57fee92c4d119e",
        "f1f81fe7467a06dc5d153cf922340a940d9d36203a2cc01c35f030276389b4ad",
    }
)

# The files this task owns; each carries the MIT header.
OWNED_FILES = (
    LINK_DIR / "__init__.py",
    LINK_DIR / "mist_link.h",
    LINK_DIR / "mist_link.cpp",
    PACKAGES / "device.yaml",
    PACKAGES / "link.yaml",
    PACKAGES / "spike.yaml",
    ENDPOINT / "spike.yaml",
    ENDPOINT / "room.yaml",
    ENDPOINT / "personal.example.yaml",
    README,
    REPO_ROOT / "scripts" / "room_endpoint" / "build.py",
)

# The player, controls and lights (plan T6); each carries the MIT header too.
PLAYER_DIR = ENDPOINT / "components" / "mist_player"
PLAYER_FILES = (
    PLAYER_DIR / "__init__.py",
    PLAYER_DIR / "mist_player.h",
    PLAYER_DIR / "mist_player.cpp",
    PACKAGES / "player.yaml",
    PACKAGES / "controls.yaml",
)
PREWAKE_DIR = ENDPOINT / "components" / "mist_prewake"

# Alarms on the device (plan T7); each carries the MIT header too.
ALARM_DIR = ENDPOINT / "components" / "mist_alarm"
ALARM_FILES = (
    ALARM_DIR / "__init__.py",
    ALARM_DIR / "mist_alarm.h",
    ALARM_DIR / "mist_alarm.cpp",
    PACKAGES / "alarm.yaml",
)

# The ids the room build's packages share.
PLAYER_ID = "mist_reply_player"
PREWAKE_ID = "mist_prewake_uplink"
# Not `alarm`: that would shadow POSIX `alarm()` in the generated C++.
ALARM_ID = "mist_alarms"

# Interfaces: the substitutions `room.yaml` carries, with the defaults the plan fixes.
ROOM_SUBSTITUTIONS = (
    "device_name",
    "wake_model",
    "stop_model",
    "vad_model",
    "voice_kit_path",
    "xmos_firmware",
    "uplink_cap_ms",
    "ring_ms",
    "backlog_ms",
    "ring_volume_floor",
)

TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


# -- The name scan -----------------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_candidates(line: str) -> Iterator[str]:
    tokens = [token for token in TOKEN_SPLIT.split(line.lower()) if token]
    yield from tokens
    for first, second in zip(tokens, tokens[1:]):
        yield f"{first}_{second}"
        yield f"{first} {second}"
        yield f"{first}{second}"


def scan_names(root: Path, digests: frozenset[str]) -> list[str]:
    """Return `<file>:<line>` for each line holding a forbidden token, never the token itself.

    Every file under `root` is read as UTF-8 (undecodable bytes replaced), lowercased and split on
    non-alphanumerics; each token, and each adjacent pair joined by `_`, a space and nothing, is
    hashed and looked up in `digests`.
    """
    hits: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        text = path.read_bytes().decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if any(_digest(candidate) in digests for candidate in _line_candidates(line)):
                hits.append(f"{path.relative_to(root).as_posix()}:{number}")
    return hits


def test_forbidden_digests_are_twelve_sha256_values() -> None:
    assert len(FORBIDDEN_DIGESTS) == 12
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in FORBIDDEN_DIGESTS)


def test_scan_reports_file_and_line_for_tokens_and_joined_pairs(tmp_path: Path) -> None:
    digests = frozenset({_digest("alpha"), _digest("beta_gamma"), _digest("deltaepsilon")})
    (tmp_path / "a.yaml").write_text("ok: 1\nmodel: ALPHA\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.cpp").write_text("// Beta-Gamma\n// delta epsilon\n// beta\n")
    hits = scan_names(tmp_path, digests)
    assert hits == ["a.yaml:2", "sub/b.cpp:1", "sub/b.cpp:2"]


def test_scan_reports_a_pair_joined_by_a_space(tmp_path: Path) -> None:
    (tmp_path / "c.md").write_text("say zeta eta now\n", encoding="utf-8")
    assert scan_names(tmp_path, frozenset({_digest("zeta eta")})) == ["c.md:1"]


def test_no_firmware_file_names_a_stock_wake_model_or_its_names() -> None:
    hits = scan_names(FIRMWARE_ROOT, FORBIDDEN_DIGESTS)
    assert not hits, f"forbidden name at {', '.join(hits)}"


# -- The YAML configs --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tagged:
    """An ESPHome tag such as `!secret` or `!include`, kept unresolved."""

    tag: str
    value: object


class _EsphomeLoader(yaml.SafeLoader):
    pass


def _construct_tagged(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Tagged:
    if isinstance(node, yaml.ScalarNode):
        return Tagged(f"!{suffix}", loader.construct_scalar(node))
    if isinstance(node, yaml.MappingNode):
        return Tagged(f"!{suffix}", loader.construct_mapping(node, deep=True))
    return Tagged(f"!{suffix}", loader.construct_sequence(node, deep=True))


_EsphomeLoader.add_multi_constructor("!", _construct_tagged)


def load(path: Path) -> dict:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_EsphomeLoader)  # nosec B506
    assert isinstance(data, dict), path
    return data


def secret(name: str) -> Tagged:
    return Tagged("!secret", name)


def include(path: str) -> Tagged:
    return Tagged("!include", path)


def test_device_package_carries_the_board_and_psram() -> None:
    device = load(PACKAGES / "device.yaml")
    esp32 = device["esp32"]
    assert esp32["variant"] == "esp32s3"
    assert esp32["flash_size"] == "16MB"
    assert esp32["framework"]["type"] == "esp-idf"
    assert esp32["framework"]["version"] == "recommended"
    assert device["psram"] == {"mode": "octal", "speed": "80MHz", "ignore_not_found": False}
    assert device["esphome"]["name"] == "${device_name}"


def test_device_package_carries_audio_buses_microphone_and_voice_kit() -> None:
    device = load(PACKAGES / "device.yaml")
    assert {bus["id"] for bus in device["i2s_audio"]} == {"i2s_input", "i2s_output"}
    (mic,) = device["microphone"]
    assert mic["platform"] == "i2s_audio"
    assert mic["sample_rate"] == 16000
    assert mic["channel"] == "stereo"
    voice_kit_sources = [
        entry for entry in device["external_components"] if entry["components"] == ["voice_kit"]
    ]
    assert voice_kit_sources == [
        {"source": {"type": "local", "path": "${voice_kit_path}"}, "components": ["voice_kit"]}
    ]
    firmware = device["voice_kit"]["firmware"]
    assert firmware["url"] == "${xmos_firmware}"
    assert firmware["md5"] == "964635c5bf125529dab14a2472a15401"


def test_device_package_takes_wifi_and_ota_from_secrets_and_time_from_sntp() -> None:
    device = load(PACKAGES / "device.yaml")
    assert device["wifi"]["ssid"] == secret("wifi_ssid")
    assert device["wifi"]["password"] == secret("wifi_password")
    (ota,) = device["ota"]
    assert ota == {"platform": "esphome", "password": secret("mist_ota_password")}
    assert [clock["platform"] for clock in device["time"]] == ["sntp"]


def test_device_package_runs_the_wake_and_stop_models_and_vad_by_substitution() -> None:
    mww = load(PACKAGES / "device.yaml")["micro_wake_word"]
    models = {model["id"]: model["model"] for model in mww["models"]}
    assert models == {"mist_wake": "${wake_model}", "stop": "${stop_model}"}
    assert mww["vad"] == {"model": "${vad_model}"}
    assert mww["microphone"]["channels"] == 1
    assert mww["microphone"]["gain_factor"] == 4


def test_link_package_takes_the_server_token_and_ca_from_the_private_directory() -> None:
    link = load(PACKAGES / "link.yaml")
    sources = link["external_components"]
    assert sources == [
        {
            "source": {"type": "local", "path": "${mist_components_path}"},
            "components": ["mist_core", "mist_link"],
        }
    ]
    mist_link = link["mist_link"]
    # An id becomes a global in the generated C++; `link` would shadow POSIX `link()`.
    assert mist_link["id"] == "mist_connection"
    assert mist_link["server_host"] == secret("mist_server_host")
    assert mist_link["server_port"] == secret("mist_server_port")
    assert mist_link["token"] == secret("mist_device_token")
    assert mist_link["ca_file"] == "mist_ca.pem"
    assert mist_link["websocket_client_path"] == "${websocket_client_path}"
    assert "device_stats" not in mist_link


def test_secret_keys_are_the_ones_devices_py_writes() -> None:
    from scripts.daily import devices

    text = (PACKAGES / "device.yaml").read_text() + (PACKAGES / "link.yaml").read_text()
    used = set(re.findall(r"!secret\s+([a-z_]+)", text))
    assert used == (set(devices.SECRET_KEYS) - {"mist_device_id"}) | {"wifi_ssid", "wifi_password"}
    assert load(PACKAGES / "link.yaml")["mist_link"]["ca_file"] == devices.DEVICE_CA_FILE


def test_spike_package_turns_on_device_stats_and_the_spike_player_only() -> None:
    assert load(PACKAGES / "spike.yaml") == {
        "mist_link": {"device_stats": True, "spike_speaker": "reply_resampling_speaker"}
    }


def test_spike_player_feeds_the_reply_resampler_of_the_speaker_chain() -> None:
    speakers = {spk["id"]: spk for spk in load(PACKAGES / "device.yaml")["speaker"]}
    resampler = speakers["reply_resampling_speaker"]
    assert resampler["platform"] == "resampler"
    assert resampler["sample_rate"] == 48000
    mixer_inputs = [source["id"] for source in speakers["mixing_speaker"]["source_speakers"]]
    assert resampler["output_speaker"] in mixer_inputs
    assert speakers["mixing_speaker"]["output_speaker"] == "i2s_audio_speaker"


def test_room_build_carries_no_spike_player() -> None:
    for path in (ENDPOINT / "room.yaml", PACKAGES / "device.yaml", PACKAGES / "link.yaml"):
        assert "spike_speaker" not in path.read_text(encoding="utf-8"), path
    for path in (*PLAYER_FILES, *ALARM_FILES):
        assert "spike_speaker" not in path.read_text(encoding="utf-8"), path


def test_spike_player_code_is_compiled_only_into_the_spike_build() -> None:
    define = "USE_MIST_LINK_SPIKE_PLAYER"
    for name in ("mist_link.h", "mist_link.cpp"):
        text = (LINK_DIR / name).read_text(encoding="utf-8")
        guarded = re.findall(rf"#ifdef {define}\n(.*?)#endif", text, flags=re.S)
        assert any("SpikePlayer" in block for block in guarded), name
        outside = re.sub(rf"#ifdef {define}\n.*?#endif", "", text, flags=re.S)
        assert "SpikePlayer" not in outside, name
    cpp = (LINK_DIR / "mist_link.cpp").read_text(encoding="utf-8")
    assert "add_playback_underruns" in cpp
    codegen = (LINK_DIR / "__init__.py").read_text(encoding="utf-8")
    assert f'cg.add_define("{define}")' in codegen
    assert "cv.use_id(speaker.Speaker)" in codegen


def test_spike_config_includes_the_spike_package() -> None:
    spike = load(ENDPOINT / "spike.yaml")
    assert spike["packages"] == {
        "device": include("packages/device.yaml"),
        "link": include("packages/link.yaml"),
        "spike": include("packages/spike.yaml"),
    }


def test_room_config_does_not_include_the_spike_package() -> None:
    room = load(ENDPOINT / "room.yaml")
    assert room["packages"] == {
        "device": include("packages/device.yaml"),
        "link": include("packages/link.yaml"),
        "player": include("packages/player.yaml"),
        "controls": include("packages/controls.yaml"),
        "alarm": include("packages/alarm.yaml"),
    }
    assert "packages/spike.yaml" not in (ENDPOINT / "room.yaml").read_text(encoding="utf-8")


def test_room_config_carries_the_interface_substitutions() -> None:
    subs = load(ENDPOINT / "room.yaml")["substitutions"]
    assert set(ROOM_SUBSTITUTIONS) <= set(subs)
    assert subs["uplink_cap_ms"] == "30000"
    assert subs["ring_ms"] == "6000"
    assert subs["backlog_ms"] == "5000"
    assert subs["ring_volume_floor"] == "0.6"
    for key in ("stop_model", "vad_model", "voice_kit_path", "xmos_firmware"):
        assert subs[key] == build.STAND_INS[key], key


def test_room_config_never_defaults_the_wake_model_to_a_real_model() -> None:
    wake = load(ENDPOINT / "room.yaml")["substitutions"]["wake_model"]
    assert wake != build.STAND_INS["stop_model"]
    assert not Path(wake).exists()
    assert "/" in wake  # a bare name would be fetched from the model registry


def test_spike_config_raises_the_uplink_cap_and_uses_stop_as_its_stand_in_wake() -> None:
    subs = load(ENDPOINT / "spike.yaml")["substitutions"]
    assert subs["uplink_cap_ms"] == "75000"
    assert subs["wake_model"] == build.STAND_INS["stop_model"]
    room = load(ENDPOINT / "room.yaml")["substitutions"]
    for key in ROOM_SUBSTITUTIONS:
        if key not in ("device_name", "wake_model", "uplink_cap_ms"):
            assert subs[key] == room[key], key


def test_personal_example_includes_the_room_config_with_placeholders() -> None:
    example = load(ENDPOINT / "personal.example.yaml")
    assert example["packages"] == {"mist": include("../room-endpoint/room.yaml")}
    assert example["substitutions"]["device_name"] == "<device>"
    assert "<" in example["substitutions"]["wake_model"]


def test_owned_files_carry_the_mit_header() -> None:
    for path in (*OWNED_FILES, *PLAYER_FILES, *ALARM_FILES):
        head = path.read_text(encoding="utf-8")[:600]
        assert "SPDX-License-Identifier: MIT" in head, path
        assert "LICENSE" in head, path


def test_owned_files_name_no_address_but_loopback() -> None:
    for path in (*OWNED_FILES, *PLAYER_FILES, *ALARM_FILES):
        addresses = set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", path.read_text("utf-8")))
        assert addresses <= {"127.0.0.1", "0.0.0.0"}, path  # nosec B104


# -- The player, controls and lights (plan T6) ------------------------------------------------


def _strings(node: object) -> list[str]:
    """Every string value under `node`, in document order: the lambdas of an action list."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, Tagged):
        return _strings(node.value)
    if isinstance(node, dict):
        return [text for value in node.values() for text in _strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _strings(value)]
    return []


def _boot_entries(path: Path) -> list[dict]:
    return load(path)["esphome"]["on_boot"]


def test_both_wake_models_are_internal_so_toggles_never_write_flash() -> None:
    models = load(PACKAGES / "device.yaml")["micro_wake_word"]["models"]
    assert [(model["id"], model.get("internal")) for model in models] == [
        ("mist_wake", True),
        ("stop", True),
    ]


def test_device_keeps_its_one_detection_handler_for_the_name_model() -> None:
    mww = load(PACKAGES / "device.yaml")["micro_wake_word"]
    assert mww["on_wake_word_detected"] == [
        {"lambda": f"id({PREWAKE_ID}).on_wake_word_detected(wake_word);"}
    ]


def test_controls_route_a_stop_detection_to_the_uplink_and_the_player() -> None:
    mww = load(PACKAGES / "controls.yaml")["micro_wake_word"]
    # The models stay in `device.yaml`; this package adds a handler only.
    assert set(mww) == {"on_wake_word_detected"}
    (handler,) = mww["on_wake_word_detected"]
    condition = handler["if"]["condition"]["lambda"].strip()
    assert condition == "return wake_word == id(stop).get_wake_word();"
    # The player answers `StopRing` while a ring plays, and the alarm component acts on it.
    assert _strings(handler["if"]["then"]) == [
        f"id({PREWAKE_ID}).on_stop_detected();\n"
        f"id({ALARM_ID}).on_ring_action(id({PLAYER_ID}).on_stop_detected());"
    ]


def test_prewake_hands_a_stop_detection_to_the_uplink() -> None:
    header = (PREWAKE_DIR / "mist_prewake.h").read_text(encoding="utf-8")
    assert "void on_stop_detected();" in header
    cpp = (PREWAKE_DIR / "mist_prewake.cpp").read_text(encoding="utf-8")
    body = re.search(r"void MistPrewake::on_stop_detected\(\) \{\n(.*?)\n\}", cpp, re.S)
    assert body is not None
    assert "this->uplink_->on_stop_keyword()" in body.group(1)


def test_controls_hand_each_button_press_to_the_player_only() -> None:
    sensors = load(PACKAGES / "controls.yaml")["binary_sensor"]
    (button,) = sensors
    assert button["platform"] == "gpio"
    assert button["id"] == "center_button"
    assert button["pin"] == {"number": "GPIO0", "inverted": True}
    calls = [_strings(click["then"]) for click in button["on_multi_click"]]
    # The player sends `interrupt` itself and answers `StopRing` or `SnoozeRing` while a ring
    # plays; the alarm component acts on those and on nothing else.
    press = f"id({PLAYER_ID}).on_button(mist_core::ButtonEvent::{{}})"
    assert calls == [
        [f"id({ALARM_ID}).on_ring_action({press.format(event)});"]
        for event in ("SinglePress", "DoublePress", "LongPress")
    ]
    # No push-to-talk: nothing the button does reaches the uplink or the wake models.
    for text in _strings(button):
        assert PREWAKE_ID not in text and "wake" not in text, text


def test_controls_turn_the_dial_into_volume_steps() -> None:
    (dial,) = load(PACKAGES / "controls.yaml")["sensor"]
    assert dial["platform"] == "rotary_encoder"
    assert (dial["pin_a"], dial["pin_b"], dial["resolution"]) == ("GPIO16", "GPIO18", 2)
    assert _strings(dial["on_clockwise"]) == [f"id({PLAYER_ID}).on_dial(1);"]
    assert _strings(dial["on_anticlockwise"]) == [f"id({PLAYER_ID}).on_dial(-1);"]


def test_controls_give_the_led_ring_an_effect_for_each_pattern_but_idle() -> None:
    lights = {light["id"]: light for light in load(PACKAGES / "controls.yaml")["light"]}
    strip = lights["leds_internal"]
    assert strip["platform"] == "esp32_rmt_led_strip"
    assert (strip["pin"], strip["num_leds"]) == ("GPIO21", 12)
    ring = lights["mist_leds"]
    assert ring["platform"] == "partition"
    assert ring["internal"] is True
    names = {next(iter(effect.values()))["name"] for effect in ring["effects"]}
    ui_state = (ENDPOINT / "components" / "mist_core" / "ui_state.h").read_text("utf-8")
    enum = re.search(r"enum class LedPattern : uint8_t \{([^}]*)\}", ui_state)
    assert enum is not None
    patterns = {name.strip() for name in enum.group(1).split(",") if name.strip()}
    # Idle turns the ring off; every other pattern is an effect the player names.
    assert names == patterns - {"Idle"}
    cpp = (PLAYER_DIR / "mist_player.cpp").read_text(encoding="utf-8")
    for name in names:
        assert f'"{name}"' in cpp, name


def test_player_feeds_the_reply_resampler_and_ducks_its_mixer_input() -> None:
    player = load(PACKAGES / "player.yaml")
    assert player["external_components"] == [
        {
            "source": {"type": "local", "path": "${mist_components_path}"},
            "components": ["mist_player"],
        }
    ]
    config = player["mist_player"]
    assert config["id"] == PLAYER_ID
    assert config["link_id"] == "mist_connection"
    assert config["speaker"] == "reply_resampling_speaker"
    assert config["ducking_speaker"] == "reply_mixing_input"
    assert config["volume_speaker"] == "i2s_audio_speaker"
    assert (config["wake_model"], config["stop_model"]) == ("mist_wake", "stop")
    assert config["mute_switch"] == "hardware_mute_switch"
    assert config["leds"] == "mist_leds"
    assert config["ring_volume_floor"] == "${ring_volume_floor}"
    speakers = {spk["id"]: spk for spk in load(PACKAGES / "device.yaml")["speaker"]}
    assert speakers["reply_resampling_speaker"]["output_speaker"] == "reply_mixing_input"
    assert speakers["i2s_audio_speaker"]["audio_dac"] == "aic3204_dac"


def test_player_sets_the_level_from_an_on_boot_after_the_device_level() -> None:
    (device_level,) = (
        entry
        for entry in _boot_entries(PACKAGES / "device.yaml")
        if any("set_volume" in text for text in _strings(entry["then"]))
    )
    assert device_level["priority"] == -100
    (boot,) = _boot_entries(PACKAGES / "player.yaml")
    # A lower priority runs later (base_automation.h, StartupTrigger).
    assert boot["priority"] < device_level["priority"]
    assert _strings(boot["then"]) == [f"id({PLAYER_ID}).apply_boot_state();"]


def test_the_player_sets_the_level_only_through_the_speakers_set_volume() -> None:
    cpp = (PLAYER_DIR / "mist_player.cpp").read_text(encoding="utf-8")
    assert re.findall(r"set_volume\(", cpp) == ["set_volume("]
    assert "this->volume_speaker_->set_volume(this->ui_.speaker_level());" in cpp


# -- Alarms on the device (plan T7) -----------------------------------------------------------


def _alarm_source() -> str:
    names = ("mist_alarm.h", "mist_alarm.cpp")
    return "".join((ALARM_DIR / name).read_text(encoding="utf-8") for name in names)


def _function_body(source: str, signature: str) -> str:
    """The body of the C++ function whose definition starts with `signature`, by brace count."""
    opening = source.index("{", source.index(signature))
    depth = 0
    for end in range(opening, len(source)):
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        if depth == 0:
            return source[opening : end + 1]
    raise AssertionError(f"unbalanced body after {signature}")


def test_alarm_package_rings_through_the_alarm_resampler_with_the_players_level() -> None:
    alarm = load(PACKAGES / "alarm.yaml")
    assert alarm["external_components"] == [
        {
            "source": {"type": "local", "path": "${mist_components_path}"},
            "components": ["mist_alarm"],
        }
    ]
    config = alarm["mist_alarm"]
    assert config == {
        "id": ALARM_ID,
        "link_id": "mist_connection",
        "player_id": PLAYER_ID,
        "speaker": "alarm_resampling_speaker",
        "time_id": "sntp_time",
        "endpoint_name": secret("mist_device_id"),
        "ring_volume_floor": "${ring_volume_floor}",
    }
    speakers = {spk["id"]: spk for spk in load(PACKAGES / "device.yaml")["speaker"]}
    assert speakers["alarm_resampling_speaker"]["platform"] == "resampler"
    assert speakers["alarm_resampling_speaker"]["output_speaker"] == "alarm_mixing_input"
    mixer_inputs = [source["id"] for source in speakers["mixing_speaker"]["source_speakers"]]
    assert "alarm_mixing_input" in mixer_inputs
    clocks = [clock["id"] for clock in load(PACKAGES / "device.yaml")["time"]]
    assert clocks == ["sntp_time"]


def test_alarm_package_names_this_device_as_the_registry_does() -> None:
    from scripts.daily import devices

    # `ring_on` lists endpoint names, and a device's is its registry name, the token's prefix.
    text = (PACKAGES / "alarm.yaml").read_text(encoding="utf-8")
    assert set(re.findall(r"!secret\s+([a-z_]+)", text)) == {"mist_device_id"}
    assert "mist_device_id" in devices.SECRET_KEYS
    assert "mist_device_id" in (REPO_ROOT / "scripts" / "room_endpoint" / "build.py").read_text(
        encoding="utf-8"
    )


def test_alarm_component_sets_the_ring_floor_and_ringing_but_never_a_volume() -> None:
    source = _alarm_source()
    assert "set_volume" not in source
    assert "set_mute" not in source
    start = _function_body(source, "void MistAlarm::start_ring(")
    assert start.index("this->player_->set_ring_floor(this->ring_floor_);") < start.index(
        "this->player_->set_ringing(true);"
    )
    stop = _function_body(source, "void MistAlarm::stop_ring(")
    assert "this->player_->set_ringing(false);" in stop
    codegen = (ALARM_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "cg.add(var.set_ring_floor(config[CONF_RING_VOLUME_FLOOR]))" in codegen


def test_alarm_component_hands_only_stop_and_snooze_to_the_rules() -> None:
    body = _function_body(_alarm_source(), "bool MistAlarm::on_ring_action(")
    assert "mist_core::ButtonAction::StopRing" in body
    assert "mist_core::ButtonAction::SnoozeRing" in body
    assert "this->rules_.on_stop(" in body
    assert "this->rules_.on_snooze(" in body
    assert "Interrupt" not in body


def test_alarm_component_listens_on_the_link_and_decodes_no_new_type() -> None:
    source = _alarm_source()
    assert "public mist_link::LinkListener" in source
    assert "this->link_->add_listener(this);" in source
    assert "read_inbound" not in source
    handled = set(re.findall(r"mist_core::InboundType::(\w+)", source))
    assert handled == {"AlarmState", "AlarmFired", "AlarmActionResult"}
    assert "mist_core::parse_alarm_state(" in source
    assert "mist_core::parse_alarm_fired(" in source


def test_alarm_component_keeps_its_memory_in_preferences_and_syncs_it() -> None:
    source = _alarm_source()
    assert re.search(r'kMemoryKey\s*=\s*"mist_alarm_schedule"', source)
    assert "global_preferences->make_preference(sizeof(mist_core::AlarmMemory)" in source
    assert "fnv1_hash(kMemoryKey)" in source
    assert "this->rules_.restore(" in source
    save = _function_body(source, "void MistAlarm::persist_(")
    assert save.index("this->rules_.save(") < save.index("this->memory_pref_.save(")
    assert "global_preferences->sync()" in save


def test_alarm_component_takes_its_time_from_sntp_else_the_owner() -> None:
    source = _alarm_source()
    assert "add_on_time_sync_callback" in source
    assert "this->clock_.on_sntp_synced();" in source
    assert "this->clock_.on_owner_time(" in source


def test_alarm_tone_is_synthesised_and_no_audio_file_ships() -> None:
    source = _alarm_source()
    assert "mist_core::AlarmTone" in source
    audio = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".pcm", ".raw"}
    assert not [path for path in ENDPOINT.rglob("*") if path.suffix.lower() in audio]
    assert "file" not in load(PACKAGES / "alarm.yaml")["mist_alarm"]


# -- The link component ------------------------------------------------------------------------


def test_link_codegen_adds_the_websocket_client_from_a_local_path_only() -> None:
    text = (LINK_DIR / "__init__.py").read_text(encoding="utf-8")
    calls = re.findall(r"add_idf_component\(([^)]*)\)", text)
    assert len(calls) == 1
    assert 'name="esp_websocket_client"' in calls[0]
    assert "path=" in calls[0]
    assert "repo=" not in calls[0] and "ref=" not in calls[0]
    assert '"/opt/mist-firmware-deps/esp_websocket_client"' in text
    assert 'include_builtin_idf_component("esp-tls")' in text
    assert 'include_builtin_idf_component("tcp_transport")' in text
    assert "request_external_task_stack()" in text


def test_link_sends_the_bearer_header_and_pins_the_ca() -> None:
    text = (LINK_DIR / "mist_link.cpp").read_text(encoding="utf-8")
    assert "Authorization: Bearer " in text
    assert "cert_pem" in text
    assert "build_voice_start" in text
    assert "/v2/device" in (LINK_DIR / "__init__.py").read_text(encoding="utf-8")


def test_link_backs_off_from_one_to_fifteen_seconds() -> None:
    header = (LINK_DIR / "mist_link.h").read_text(encoding="utf-8")
    assert re.search(r"kBackoffFirstMs\s*=\s*1000\b", header)
    assert re.search(r"kBackoffMaxMs\s*=\s*15000\b", header)


def test_link_sender_task_stack_is_in_psram() -> None:
    text = (LINK_DIR / "mist_link.cpp").read_text(encoding="utf-8")
    assert re.search(r"sender_task_\.create\([^;]*,\s*true\)", text, re.S)


def test_device_stats_code_is_compiled_only_into_the_spike_build() -> None:
    text = (LINK_DIR / "mist_link.cpp").read_text(encoding="utf-8")
    guarded = re.findall(r"#ifdef USE_MIST_LINK_DEVICE_STATS\n(.*?)#endif", text, flags=re.S)
    assert any("build_device_stats" in block for block in guarded)
    outside = re.sub(r"#ifdef USE_MIST_LINK_DEVICE_STATS\n.*?#endif", "", text, flags=re.S)
    assert "build_device_stats" not in outside
    codegen = (LINK_DIR / "__init__.py").read_text(encoding="utf-8")
    assert 'cg.add_define("USE_MIST_LINK_DEVICE_STATS")' in codegen


# -- build.py ----------------------------------------------------------------------------------


@dataclass
class FakeToolchain:
    """Records what `build.py` asks of the compile, the server and the CA maker."""

    compile_rc: int = 0
    write_images: bool = True
    compiles: list[tuple[list[str], dict[str, str], Path]] = field(default_factory=list)
    served: list[list[str]] = field(default_factory=list)
    serving_during_compile: list[bool] = field(default_factory=list)
    seen_files: dict[str, str] = field(default_factory=dict)
    stop_manifest: Path = Path(build.STAND_INS["stop_model"])
    _serving: bool = False

    def compile(self, argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
        self.compiles.append((list(argv), dict(env), cwd))
        self.serving_during_compile.append(self._serving)
        for path in cwd.rglob("*"):
            if path.is_file() and path.suffix in (".yaml", ".pem", ".json"):
                self.seen_files[path.relative_to(cwd).as_posix()] = path.read_text("utf-8")
        if self.write_images:
            images = Path(env["ESPHOME_DATA_DIR"]) / "build" / "mist-test" / "build"
            images.mkdir(parents=True)
            (images / build.FACTORY_IMAGE).write_bytes(b"factory")
            (images / build.OTA_IMAGE).write_bytes(b"ota")
        return self.compile_rc

    @contextmanager
    def serve(self, argv: Sequence[str]) -> Iterator[None]:
        self.served.append(list(argv))
        self._serving = True
        try:
            yield
        finally:
            self._serving = False

    def make_ca(self, pem: Path, work_dir: Path) -> None:
        pem.write_text("-----BEGIN CERTIFICATE-----\nstand-in\n-----END CERTIFICATE-----\n")

    def toolchain(self, tmp_root: Path) -> build.Toolchain:
        return build.Toolchain(
            compile=self.compile,
            serve=self.serve,
            make_ca=self.make_ca,
            tmp_root=tmp_root,
            stop_manifest=self.stop_manifest,
        )


@pytest.fixture
def stop_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The image's stop manifest layout, outside the build root the test checks is emptied."""
    stop = tmp_path_factory.mktemp("deps") / "models" / "stop"
    stop.mkdir(parents=True)
    (stop / "stop.tflite").write_bytes(b"stand-in model")
    manifest = {
        "type": "micro",
        "wake_word": "Stop",
        "model": "stop.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {"probability_cutoff": 0.5, "sliding_window_size": 5},
    }
    (stop / "stop.json").write_text(json.dumps(manifest), encoding="utf-8")
    return stop / "stop.json"


def _substitutions(argv: Sequence[str]) -> dict[str, str]:
    subs = dict(zip(argv[2:-2:3], argv[3:-2:3]))
    assert argv[1:-2:3] == ["-s"] * len(subs)
    return subs


def _run(fake: FakeToolchain, tmp_root: Path, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    rc = build.main(list(args), toolchain=fake.toolchain(tmp_root), stdout=out)
    return rc, out.getvalue()


def test_compile_argv_puts_each_substitution_before_the_command() -> None:
    argv = build.compile_argv(Path("/tmp/b/room-endpoint/spike.yaml"), {"a": "1", "b": "two"})
    assert argv == [
        "esphome",
        "-s",
        "a",
        "1",
        "-s",
        "b",
        "two",
        "compile",
        "/tmp/b/room-endpoint/spike.yaml",
    ]


def test_server_serves_the_deps_on_loopback_port_8765() -> None:
    assert build.server_argv("python3.12") == [
        "python3.12",
        "-I",
        "-m",
        "http.server",
        "8765",
        "--bind",
        "127.0.0.1",
        "--directory",
        "/opt/mist-firmware-deps",
    ]
    assert build.STAND_INS["xmos_firmware"] == (
        "http://127.0.0.1:8765/xmos/ffva_v1.3.1_upgrade.bin"
    )


def test_environment_puts_data_dir_and_home_in_the_build_directory(tmp_path: Path) -> None:
    env = build.build_env(tmp_path, {"PATH": "/usr/bin", "HOME": "/root", "ESPHOME_DATA_DIR": "x"})
    assert env == {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path / "home"),
        "ESPHOME_DATA_DIR": str(tmp_path / "data"),
    }


def test_worker_compile_copies_the_tree_and_passes_the_stand_ins(tmp_path: Path) -> None:
    fake = FakeToolchain()
    rc, output = _run(fake, tmp_path, "compile", "spike")
    assert rc == 0, output
    ((argv, env, cwd),) = fake.compiles
    assert cwd.parent == tmp_path
    tree = cwd / "room-endpoint"
    assert argv[-2:] == ["compile", str(tree / "spike.yaml")]
    subs = _substitutions(argv)
    assert subs == {**build.STAND_INS, "mist_components_path": str(tree / "components")}
    assert env["ESPHOME_DATA_DIR"] == str(cwd / "data")
    assert env["HOME"] == str(cwd / "home")
    copied = (ENDPOINT / "spike.yaml").read_text(encoding="utf-8")
    assert fake.seen_files["room-endpoint/spike.yaml"] == copied
    assert "room-endpoint/packages/link.yaml" in fake.seen_files
    assert fake.served == [build.server_argv(build.server_python())]
    assert fake.serving_during_compile == [True]
    assert list(tmp_path.iterdir()) == []  # the build directory is removed


def test_worker_compile_writes_throwaway_secrets_in_the_build_directory(
    tmp_path: Path, stop_manifest: Path
) -> None:
    fake = FakeToolchain(stop_manifest=stop_manifest)
    rc, _ = _run(fake, tmp_path, "compile", "room")
    assert rc == 0
    secrets_text = fake.seen_files["room-endpoint/secrets.yaml"]
    values = yaml.safe_load(secrets_text)
    assert set(values) == {
        "wifi_ssid",
        "wifi_password",
        "mist_device_id",
        "mist_device_token",
        "mist_server_host",
        "mist_server_port",
        "mist_ota_password",
    }
    assert values["mist_server_host"] == "127.0.0.1"
    assert values["mist_server_port"] == 8443
    device, _, secret_part = values["mist_device_token"].partition(".")
    assert device == values["mist_device_id"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", secret_part)
    assert len(values["wifi_password"]) >= 8
    assert fake.seen_files["room-endpoint/mist_ca.pem"].startswith("-----BEGIN CERTIFICATE-----")
    assert not (ENDPOINT / "secrets.yaml").exists()
    assert not (ENDPOINT / "mist_ca.pem").exists()


def test_without_out_the_images_are_printed_and_discarded(tmp_path: Path) -> None:
    fake = FakeToolchain()
    rc, output = _run(fake, tmp_path, "compile", "spike")
    assert rc == 0
    assert re.search(r"^factory image: \S+/firmware\.factory\.bin", output, re.M)
    assert re.search(r"^ota image: \S+/firmware\.ota\.bin", output, re.M)
    assert "discarded" in output
    assert list(tmp_path.iterdir()) == []


def test_out_copies_both_images_and_prints_their_paths(tmp_path: Path) -> None:
    fake = FakeToolchain()
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    out = tmp_path / "private" / "images" / "spike"
    rc, output = _run(fake, tmp_root, "compile", "spike", "--out", str(out))
    assert rc == 0
    assert (out / "firmware.factory.bin").read_bytes() == b"factory"
    assert (out / "firmware.ota.bin").read_bytes() == b"ota"
    assert f"factory image: {out / 'firmware.factory.bin'}" in output
    assert f"ota image: {out / 'firmware.ota.bin'}" in output
    assert list(tmp_root.iterdir()) == []


def test_config_builds_the_private_yaml_beside_a_copy_of_the_tree(tmp_path: Path) -> None:
    private = tmp_path / "private" / "device"
    (private / "images" / "spike").mkdir(parents=True)
    (private / "images" / "spike" / "old.bin").write_bytes(b"old")
    (private / "spike.yaml").write_text("packages:\n  mist: !include ../room-endpoint/spike.yaml\n")
    (private / "secrets.yaml").write_text("wifi_ssid: private\n")
    (private / "mist_ca.pem").write_text("-----BEGIN CERTIFICATE-----\nprivate\n")
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    fake = FakeToolchain()
    rc, output = _run(
        fake,
        tmp_root,
        "compile",
        "spike",
        "--config",
        str(private / "spike.yaml"),
        "--out",
        str(private / "images" / "spike"),
    )
    assert rc == 0, output
    ((argv, _, cwd),) = fake.compiles
    assert argv == build.compile_argv(
        cwd / "config" / "spike.yaml",
        {"mist_components_path": str(cwd / "room-endpoint" / "components")},
    )
    assert fake.seen_files["config/secrets.yaml"] == "wifi_ssid: private\n"
    assert "room-endpoint/secrets.yaml" not in fake.seen_files
    assert not (cwd / "config" / "images").exists()
    assert (private / "images" / "spike" / "firmware.factory.bin").read_bytes() == b"factory"


def test_room_compile_check_wakes_on_a_stand_in_word_in_the_build_directory(
    tmp_path: Path, stop_manifest: Path
) -> None:
    fake = FakeToolchain(stop_manifest=stop_manifest)
    rc, output = _run(fake, tmp_path, "compile", "room")
    assert rc == 0, output
    ((argv, _, cwd),) = fake.compiles
    subs = _substitutions(argv)
    wake = Path(subs["wake_model"])
    rel = wake.relative_to(cwd).as_posix()  # under the throwaway build directory
    assert wake.suffix == ".json"
    assert subs == {
        **build.STAND_INS,
        "wake_model": str(wake),
        "mist_components_path": str(cwd / "room-endpoint" / "components"),
    }
    written = json.loads(fake.seen_files[rel])
    stop = json.loads(stop_manifest.read_text(encoding="utf-8"))
    assert written["wake_word"] == "compile check"
    assert written["wake_word"].strip().lower() != stop["wake_word"].strip().lower()
    unchanged = {k: v for k, v in stop.items() if k not in ("wake_word", "model")}
    assert {k: v for k, v in written.items() if k not in ("wake_word", "model")} == unchanged
    # ESPHome loads `manifest.parent / manifest["model"]`; an absolute `model` replaces the parent.
    assert Path(written["model"]).is_absolute()
    assert wake.parent / written["model"] == stop_manifest.resolve().parent / stop["model"]
    assert list(tmp_path.iterdir()) == []


def test_the_room_stand_in_wake_copies_the_image_stop_manifest() -> None:
    assert build.default_toolchain().stop_manifest == Path(build.STAND_INS["stop_model"])
    assert build.STAND_INS["stop_model"] == "/opt/mist-firmware-deps/models/stop/stop.json"


def test_spike_compile_keeps_the_stop_model_as_its_wake(tmp_path: Path) -> None:
    fake = FakeToolchain(stop_manifest=tmp_path / "missing" / "stop.json")
    rc, output = _run(fake, tmp_path, "compile", "spike")
    assert rc == 0, output
    ((argv, _, _),) = fake.compiles
    assert _substitutions(argv)["wake_model"] == build.STAND_INS["stop_model"]
    assert not any(name.endswith(".json") for name in fake.seen_files)


def test_room_compile_check_refuses_out_before_any_compile(tmp_path: Path) -> None:
    fake = FakeToolchain()
    out = tmp_path / "out"
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    rc, output = _run(fake, tmp_root, "compile", "room", "--out", str(out))
    assert rc == 2
    assert output.startswith("refused: ")
    assert "--out" in output and "--config" in output
    assert fake.compiles == []
    assert fake.served == []
    assert not out.exists()
    assert list(tmp_root.iterdir()) == []


def test_room_with_config_keeps_out(tmp_path: Path) -> None:
    private = tmp_path / "private" / "device"
    private.mkdir(parents=True)
    (private / "room.yaml").write_text("packages:\n  mist: !include ../room-endpoint/room.yaml\n")
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    out = private / "images" / "room"
    fake = FakeToolchain()
    rc, output = _run(
        fake, tmp_root, "compile", "room", "--config", str(private / "room.yaml"), "--out", str(out)
    )
    assert rc == 0, output
    ((argv, _, cwd),) = fake.compiles
    assert argv == build.compile_argv(
        cwd / "config" / "room.yaml",
        {"mist_components_path": str(cwd / "room-endpoint" / "components")},
    )
    assert (out / "firmware.factory.bin").read_bytes() == b"factory"
    assert (out / "firmware.ota.bin").read_bytes() == b"ota"


def test_a_failed_compile_exits_1_and_removes_the_build_directory(tmp_path: Path) -> None:
    fake = FakeToolchain(compile_rc=2, write_images=False)
    out = tmp_path / "out"
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    rc, output = _run(fake, tmp_root, "compile", "spike", "--out", str(out))
    assert rc == 1
    assert "compile failed" in output
    assert not out.exists()
    assert list(tmp_root.iterdir()) == []


def test_a_missing_image_exits_1(tmp_path: Path) -> None:
    fake = FakeToolchain(write_images=False)
    rc, output = _run(fake, tmp_path, "compile", "spike")
    assert rc == 1
    assert "firmware.factory.bin" in output


def test_a_missing_config_is_refused_with_exit_2(tmp_path: Path) -> None:
    fake = FakeToolchain()
    rc, output = _run(fake, tmp_path, "compile", "spike", "--config", str(tmp_path / "no.yaml"))
    assert rc == 2
    assert fake.compiles == []


def test_the_real_ca_maker_writes_a_certificate(tmp_path: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not on PATH")
    pem = tmp_path / "mist_ca.pem"
    build.make_stand_in_ca(pem, tmp_path)
    text = pem.read_text(encoding="ascii")
    assert text.startswith("-----BEGIN CERTIFICATE-----")
    assert "PRIVATE" not in text


def test_build_module_is_the_one_the_plan_names() -> None:
    assert build.__name__ == "scripts.room_endpoint.build"


# -- The README --------------------------------------------------------------------------------


def _readme_commands(prefix: str) -> list[list[str]]:
    text = README.read_text(encoding="utf-8").replace("\\\n", " ")
    commands = []
    for line in text.splitlines():
        if prefix in line:
            commands.append(shlex.split(line.split(prefix, 1)[1]))
    return commands


def test_readme_build_command_parses_with_build_py() -> None:
    commands = _readme_commands("python -m scripts.room_endpoint.build ")
    assert commands, "the README gives no build command"
    parser = build.build_parser()
    parsed = [parser.parse_args(args) for args in commands]
    assert any(
        args.command == "compile"
        and args.target == "spike"
        and args.config == Path("/private/<device>/spike.yaml")
        and args.out == Path("/private/<device>/images/spike")
        for args in parsed
    ), "the README gives no H4 spike build command"


def test_readme_first_section_gives_the_h4_steps() -> None:
    text = README.read_text(encoding="utf-8")
    first = text.split("\n## ", 2)[1]
    for needle in (
        "python -m scripts.room_endpoint.build compile spike",
        "web.esphome.io",
        "python -m scripts.room_endpoint.spike_listener",
        "-p <PC LAN IP>:8443:8443",
        "--network none",
        "tone",
        "barge_in",
    ):
        assert needle in first, needle
    for command in ("state", "reject", "tone", "status", "quit"):
        assert f"`{command}" in first, command
    flat = " ".join(first.split())
    assert 'say "stop" with nothing playing (it opens the uplink), then type `tone` at once' in flat
