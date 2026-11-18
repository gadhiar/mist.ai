# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""ESPHome codegen for `mist_link`, the room endpoint's one link to MIST's device listener.

The link speaks protocol 2 over `wss://<server_host>:<server_port>/v2/device` through
`esp_websocket_client` (Apache-2.0), with the private CA pinned and `Authorization: Bearer
<token>`. The client is added as an IDF component from the worker image's local copy, never by
registry name, so an offline compile has nothing to fetch; `esp-tls` and `tcp_transport`, which
ESPHome leaves out of the build by default, are put back for it.

`device_stats: true` (the spike build's `packages/spike.yaml`) compiles in the 10 s stats report;
without it the report's code is not in the image. `spike_speaker` (also the spike build's only)
compiles in a minimal player that sends `0x01` audio straight to that speaker, so the spike can
measure uplink during playback before the room build's player exists; it has no tag rules, fades
or ducking.
"""

from pathlib import Path

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import psram, speaker
from esphome.components.esp32 import (
    add_idf_component,
    add_idf_sdkconfig_option,
    include_builtin_idf_component,
)
from esphome.const import CONF_ID

CODEOWNERS: list[str] = []
DEPENDENCIES = ["esp32", "psram", "wifi"]
AUTO_LOAD = ["mist_core"]

CONF_SERVER_HOST = "server_host"
CONF_SERVER_PORT = "server_port"
CONF_TOKEN = "token"  # nosec B105 - a config key's name, not a secret
CONF_CA_FILE = "ca_file"
CONF_WEBSOCKET_CLIENT_PATH = "websocket_client_path"
CONF_DEVICE_STATS = "device_stats"
CONF_SPIKE_SPEAKER = "spike_speaker"

DEFAULT_WEBSOCKET_CLIENT_PATH = "/opt/mist-firmware-deps/esp_websocket_client"
DEVICE_PATH = "/v2/device"
PEM_BEGIN = "-----BEGIN CERTIFICATE-----"

mist_link_ns = cg.esphome_ns.namespace("mist_link")
MistLink = mist_link_ns.class_("MistLink", cg.Component)


def _token(value: object) -> str:
    """A bearer token is `<device>.<secret>`, as `scripts/daily/devices.py` writes it."""
    text = cv.string_strict(value)
    device, sep, secret = text.partition(".")
    if not sep or not device or not secret:
        raise cv.Invalid("the token must be <device>.<secret>, as devices.py add writes it")
    return text


def _ca_file(value: object) -> Path:
    """The pinned CA certificate, read relative to the top-level YAML's directory."""
    path = cv.file_(value)
    if PEM_BEGIN not in path.read_text(encoding="ascii", errors="replace"):
        raise cv.Invalid(f"{path} holds no PEM certificate")
    return path


CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(MistLink),
        cv.Required(CONF_SERVER_HOST): cv.string_strict,
        cv.Optional(CONF_SERVER_PORT, default=8443): cv.port,
        cv.Required(CONF_TOKEN): _token,
        cv.Required(CONF_CA_FILE): _ca_file,
        cv.Optional(
            CONF_WEBSOCKET_CLIENT_PATH, default=DEFAULT_WEBSOCKET_CLIENT_PATH
        ): cv.directory,
        cv.Optional(CONF_DEVICE_STATS, default=False): cv.boolean,
        cv.Optional(CONF_SPIKE_SPEAKER): cv.use_id(speaker.Speaker),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config: dict[str, object]) -> None:
    """Declare the link and add the WebSocket client and the TLS pieces it needs."""
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    uri = f"wss://{config[CONF_SERVER_HOST]}:{config[CONF_SERVER_PORT]}{DEVICE_PATH}"
    cg.add(var.set_uri(uri))
    cg.add(var.set_token(config[CONF_TOKEN]))
    cg.add(var.set_ca_pem(Path(config[CONF_CA_FILE]).read_text(encoding="ascii")))

    include_builtin_idf_component("esp-tls")
    include_builtin_idf_component("tcp_transport")
    add_idf_component(name="esp_websocket_client", path=str(config[CONF_WEBSOCKET_CLIENT_PATH]))
    # Sends from the sender task do not wait on the client task's receive lock.
    add_idf_sdkconfig_option("CONFIG_ESP_WS_CLIENT_SEPARATE_TX_LOCK", True)
    # The sender task's stack lives in PSRAM.
    psram.request_external_task_stack()

    if config[CONF_DEVICE_STATS]:
        cg.add_define("USE_MIST_LINK_DEVICE_STATS")
    if CONF_SPIKE_SPEAKER in config:
        cg.add_define("USE_MIST_LINK_SPIKE_PLAYER")
        output = await cg.get_variable(config[CONF_SPIKE_SPEAKER])
        cg.add(var.set_spike_speaker(output))
