# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""ESPHome codegen for `mist_player`, the room build's player, controls and lights.

The player is a `mist_link` listener: it plays the `0x01` audio of the turn `stream_start` names
through `speaker` (the reply resampler), ducks `ducking_speaker` (the reply's mixer input), and
sets the output level only through `volume_speaker`'s `set_volume` (the I2S speaker, which hands
it to the DAC). The name and stop models, the LED ring and the level follow `mist_core`'s
`UiState`; `ring_volume_floor` is the quietest a ring plays. The inputs are wired in
`packages/controls.yaml`, and the boot level in `packages/player.yaml`.
"""

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor, light, speaker
from esphome.components.micro_wake_word import WakeWordModel
from esphome.components.mist_link import MistLink
from esphome.components.mixer.speaker import SourceSpeaker
from esphome.const import CONF_ID, CONF_SPEAKER

CODEOWNERS: list[str] = []
DEPENDENCIES = ["esp32", "psram", "mist_link", "micro_wake_word", "speaker", "light"]
AUTO_LOAD = ["mist_core"]

CONF_LINK_ID = "link_id"
CONF_DUCKING_SPEAKER = "ducking_speaker"
CONF_VOLUME_SPEAKER = "volume_speaker"
CONF_WAKE_MODEL = "wake_model"
CONF_STOP_MODEL = "stop_model"
CONF_MUTE_SWITCH = "mute_switch"
CONF_LEDS = "leds"
CONF_RING_VOLUME_FLOOR = "ring_volume_floor"

mist_player_ns = cg.esphome_ns.namespace("mist_player")
MistPlayer = mist_player_ns.class_("MistPlayer", cg.Component)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(MistPlayer),
        cv.GenerateID(CONF_LINK_ID): cv.use_id(MistLink),
        cv.Required(CONF_SPEAKER): cv.use_id(speaker.Speaker),
        cv.Required(CONF_DUCKING_SPEAKER): cv.use_id(SourceSpeaker),
        cv.Required(CONF_VOLUME_SPEAKER): cv.use_id(speaker.Speaker),
        cv.Required(CONF_WAKE_MODEL): cv.use_id(WakeWordModel),
        cv.Required(CONF_STOP_MODEL): cv.use_id(WakeWordModel),
        cv.Optional(CONF_MUTE_SWITCH): cv.use_id(binary_sensor.BinarySensor),
        cv.Required(CONF_LEDS): cv.use_id(light.LightState),
        cv.Optional(CONF_RING_VOLUME_FLOOR, default=0.6): cv.float_range(min=0.0, max=1.0),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config: dict) -> None:
    """Declare the player and hand it the link, speakers, models, switch and LEDs."""
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_link(await cg.get_variable(config[CONF_LINK_ID])))
    cg.add(var.set_speaker(await cg.get_variable(config[CONF_SPEAKER])))
    cg.add(var.set_ducking_speaker(await cg.get_variable(config[CONF_DUCKING_SPEAKER])))
    cg.add(var.set_volume_speaker(await cg.get_variable(config[CONF_VOLUME_SPEAKER])))
    cg.add(var.set_wake_model(await cg.get_variable(config[CONF_WAKE_MODEL])))
    cg.add(var.set_stop_model(await cg.get_variable(config[CONF_STOP_MODEL])))
    if CONF_MUTE_SWITCH in config:
        cg.add(var.set_mute_switch(await cg.get_variable(config[CONF_MUTE_SWITCH])))
    cg.add(var.set_leds(await cg.get_variable(config[CONF_LEDS])))
    cg.add(var.set_ring_floor(config[CONF_RING_VOLUME_FLOOR]))
