# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""ESPHome codegen for `mist_prewake`, the room endpoint's pre-wake ring and wake uplink.

A `MicrophoneSource` tap (channel 0, gain 1) fills a `ring_ms` ring in PSRAM; a detection whose
word is `wake_model`'s opens the uplink through `mist_link`; the mute switch clears the ring,
closes any uplink and sends `mute`. The state machine is `mist_core`'s `Uplink`.

`micro_wake_word`'s detection trigger passes only the detected model's word, so the component
tells models apart by word. The final validation below makes that safe: no other model may share
the name model's word, except in the spike build, whose stand-in wake is the stop model itself.
"""

import json
from pathlib import Path

import esphome.codegen as cg
import esphome.config_validation as cv
import esphome.final_validate as fv
from esphome.components import binary_sensor, microphone, network
from esphome.components.micro_wake_word import WakeWordModel
from esphome.components.mist_link import MistLink
from esphome.const import CONF_ID, CONF_MICROPHONE, CONF_PATH, CONF_TYPE, TYPE_LOCAL

CODEOWNERS: list[str] = []
DEPENDENCIES = ["esp32", "psram", "mist_link", "micro_wake_word", "microphone"]
AUTO_LOAD = ["mist_core"]

CONF_LINK_ID = "link_id"
CONF_WAKE_MODEL = "wake_model"
CONF_MUTE_SWITCH = "mute_switch"
CONF_RING_MS = "ring_ms"
CONF_BACKLOG_MS = "backlog_ms"
CONF_UPLINK_CAP_MS = "uplink_cap_ms"
CONF_DEVICE_STATS = "device_stats"
CONF_MICRO_WAKE_WORD = "micro_wake_word"
CONF_MIST_LINK = "mist_link"
CONF_MODEL = "model"
CONF_MODELS = "models"
KEY_WAKE_WORD = "wake_word"

FRAME_MS = 20
MAX_BACKLOG_MS = 5000

mist_prewake_ns = cg.esphome_ns.namespace("mist_prewake")
MistPrewake = mist_prewake_ns.class_("MistPrewake", cg.Component)


def _ring_holds_the_backlog(config: dict) -> dict:
    """The ring keeps one frame more than the backlog, so a wake can send the whole backlog."""
    if config[CONF_RING_MS] < config[CONF_BACKLOG_MS] + FRAME_MS:
        raise cv.Invalid(f"{CONF_RING_MS} must exceed {CONF_BACKLOG_MS} by at least {FRAME_MS} ms")
    return config


def _request_high_performance_networking(config: dict) -> dict:
    """Size lwIP and Wi-Fi for the backlog burst, as ESPHome's streaming components do.

    A wake sends 5 s of backlog, about 175 KB on the wire, that must arrive within 1 s. Without
    this request lwIP keeps ESP-IDF's 5760-byte send buffer (`lwip/Kconfig:652-654`), so about 30
    round trips carry one backlog. With PSRAM the request gives a 65534-byte send buffer with
    window scaling (`network/__init__.py:438-469`) and more Wi-Fi buffers with A-MPDU, allocated
    in PSRAM (`wifi/__init__.py:785-814`). The published Voice PE config gets the same through
    Sendspin (`sendspin/__init__.py:252-258`). Only this request reaches the Wi-Fi settings:
    `network: enable_high_performance` sets lwIP's alone (`has_high_performance_networking`,
    `network/__init__.py:201-215`). It runs at validation, before any `to_code` reads the flag.
    """
    network.require_high_performance_networking()
    return config


CONFIG_SCHEMA = cv.All(
    cv.Schema(
        {
            cv.GenerateID(): cv.declare_id(MistPrewake),
            cv.GenerateID(CONF_LINK_ID): cv.use_id(MistLink),
            cv.Required(CONF_MICROPHONE): microphone.microphone_source_schema(
                min_bits_per_sample=16,
                max_bits_per_sample=16,
                min_channels=1,
                max_channels=1,
            ),
            cv.Required(CONF_WAKE_MODEL): cv.use_id(WakeWordModel),
            cv.Optional(CONF_MUTE_SWITCH): cv.use_id(binary_sensor.BinarySensor),
            cv.Optional(CONF_RING_MS, default=6000): cv.int_range(min=2 * FRAME_MS, max=60000),
            cv.Optional(CONF_BACKLOG_MS, default=MAX_BACKLOG_MS): cv.int_range(
                min=0, max=MAX_BACKLOG_MS
            ),
            cv.Optional(CONF_UPLINK_CAP_MS, default=30000): cv.int_range(min=1000, max=600000),
        }
    ).extend(cv.COMPONENT_SCHEMA),
    _ring_holds_the_backlog,
    _request_high_performance_networking,
)


def _wake_word(model_parameters: dict) -> tuple[str, Path]:
    """The manifest's word and path of one `micro_wake_word` model; local models only."""
    model = model_parameters[CONF_MODEL]
    if model[CONF_TYPE] != TYPE_LOCAL:
        raise cv.Invalid("mist_prewake reads local wake models only; the build fetches nothing")
    path = Path(model[CONF_PATH]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return str(manifest[KEY_WAKE_WORD]).strip().lower(), path


def _only_the_name_model_opens(config: dict) -> dict:
    """Refuse a build where a model other than `wake_model` could open the uplink.

    The component opens on a detection whose word is the name model's, so no other model may
    carry that word. The one exception is the spike build's stand-in wake (plan Decision 6a):
    there `wake_model` is the stop model's own file. `mist_link: device_stats` marks the spike
    build, which only `packages/spike.yaml` sets, so the room build can never open on stop.
    """
    full = fv.full_config.get()
    models = full[CONF_MICRO_WAKE_WORD][CONF_MODELS]
    wake_id = config[CONF_WAKE_MODEL]
    names = [m for m in models if m[CONF_ID].id == wake_id.id]
    if len(names) != 1:
        raise cv.Invalid(f"{CONF_WAKE_MODEL} must name one micro_wake_word model")
    name_word, name_path = _wake_word(names[0])
    spike = bool(full.get(CONF_MIST_LINK, {}).get(CONF_DEVICE_STATS, False))
    for other in models:
        if other is names[0]:
            continue
        word, path = _wake_word(other)
        if word != name_word:
            continue
        if spike and path == name_path:
            continue
        raise cv.Invalid(
            f"model '{other[CONF_ID].id}' shares the wake word of {CONF_WAKE_MODEL}, so it would "
            "open the uplink; only the spike build may use the stop model as its stand-in wake"
        )
    return config


FINAL_VALIDATE_SCHEMA = cv.All(
    cv.Schema(
        {
            cv.Required(CONF_MICROPHONE): microphone.final_validate_microphone_source_schema(
                "mist_prewake", sample_rate=16000
            ),
        },
        extra=cv.ALLOW_EXTRA,
    ),
    _only_the_name_model_opens,
)


async def to_code(config: dict) -> None:
    """Declare the tap and uplink, and wire them to the link, microphone, model and switch."""
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_link(await cg.get_variable(config[CONF_LINK_ID])))
    mic_source = await microphone.microphone_source_to_code(config[CONF_MICROPHONE])
    cg.add(var.set_microphone_source(mic_source))
    cg.add(var.set_wake_model(await cg.get_variable(config[CONF_WAKE_MODEL])))
    if CONF_MUTE_SWITCH in config:
        cg.add(var.set_mute_switch(await cg.get_variable(config[CONF_MUTE_SWITCH])))
    cg.add(var.set_ring_ms(config[CONF_RING_MS]))
    cg.add(var.set_backlog_ms(config[CONF_BACKLOG_MS]))
    cg.add(var.set_uplink_cap_ms(config[CONF_UPLINK_CAP_MS]))
