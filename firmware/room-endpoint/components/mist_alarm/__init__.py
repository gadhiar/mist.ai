# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""ESPHome codegen for `mist_alarm`, MIST's alarms on the room device (plan Design item 6, T7).

The component is a `mist_link` listener for `alarm_state`, `alarm_fired` and
`alarm_action_result`, which `mist_core`'s alarm rules act on. A ring plays a tone synthesised
on the device through `speaker` (the alarm resampler, which feeds its own mixer input), while the
player (`player_id`) owns the level: before each ring the component hands it `ring_volume_floor`
and turns its ringing on, so the LEDs, the stop model and the level follow. The component never
sets a volume itself.

`endpoint_name` is this device's name as the server's registry knows it, which `ring_on` lists;
`scripts/daily/devices.py add` writes it to `secrets.yaml` as `mist_device_id`. `time_id` is the
SNTP clock the rules run by once it has synced; until then the owner's own clock, from each
`alarm_state.at`, stands in.
"""

import re

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import speaker
from esphome.components import time as time_
from esphome.components.mist_link import MistLink
from esphome.components.mist_player import MistPlayer
from esphome.const import CONF_ID, CONF_SPEAKER, CONF_TIME_ID

CODEOWNERS: list[str] = []
DEPENDENCIES = ["esp32", "psram", "mist_link", "mist_player", "speaker", "time"]
AUTO_LOAD = ["mist_core"]

CONF_LINK_ID = "link_id"
CONF_PLAYER_ID = "player_id"
CONF_ENDPOINT_NAME = "endpoint_name"
CONF_RING_VOLUME_FLOOR = "ring_volume_floor"

# The registry's own rule for a device name (`backend/devices/registry.py`, NAME_PATTERN and
# RESERVED_NAMES), so a name the server would never list in `ring_on` fails the compile.
ENDPOINT_NAME = re.compile(r"[a-z][a-z0-9-]{0,23}", re.ASCII)
RESERVED_NAMES = frozenset({"desktop", "all", "room"})

mist_alarm_ns = cg.esphome_ns.namespace("mist_alarm")
MistAlarm = mist_alarm_ns.class_("MistAlarm", cg.Component)


def _endpoint_name(value: object) -> str:
    """A device name the registry accepts, as `devices.py add` writes it."""
    text = cv.string_strict(value)
    if ENDPOINT_NAME.fullmatch(text) is None or text in RESERVED_NAMES:
        raise cv.Invalid(
            "endpoint_name must be the device's registry name: a lowercase letter, then up to 23 "
            "lowercase letters, digits or hyphens, and not desktop, all or room"
        )
    return text


CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(MistAlarm),
        cv.GenerateID(CONF_LINK_ID): cv.use_id(MistLink),
        cv.GenerateID(CONF_PLAYER_ID): cv.use_id(MistPlayer),
        cv.Required(CONF_SPEAKER): cv.use_id(speaker.Speaker),
        cv.GenerateID(CONF_TIME_ID): cv.use_id(time_.RealTimeClock),
        cv.Required(CONF_ENDPOINT_NAME): _endpoint_name,
        cv.Optional(CONF_RING_VOLUME_FLOOR, default=0.6): cv.float_range(min=0.0, max=1.0),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config: dict) -> None:
    """Declare the alarm component and hand it the link, player, speaker, clock and name."""
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_link(await cg.get_variable(config[CONF_LINK_ID])))
    cg.add(var.set_player(await cg.get_variable(config[CONF_PLAYER_ID])))
    cg.add(var.set_speaker(await cg.get_variable(config[CONF_SPEAKER])))
    cg.add(var.set_time(await cg.get_variable(config[CONF_TIME_ID])))
    cg.add(var.set_endpoint_name(config[CONF_ENDPOINT_NAME]))
    cg.add(var.set_ring_floor(config[CONF_RING_VOLUME_FLOOR]))
