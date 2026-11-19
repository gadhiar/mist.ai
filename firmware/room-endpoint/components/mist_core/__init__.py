# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""ESPHome descriptor for `mist_core`, the room endpoint's portable core.

The core is plain C++17 with no ESPHome or IDF header, so the host tests build it with g++
(`tests/unit/firmware`). This descriptor lets the device's other components load it as a
library: it has no configuration of its own, and its sources compile with whichever component
names it in `AUTO_LOAD`.
"""

import esphome.codegen as cg
import esphome.config_validation as cv

CODEOWNERS: list[str] = []

CONFIG_SCHEMA = cv.Schema({})


async def to_code(config: dict[str, object]) -> None:
    """Mark the core as present for the components that include it."""
    cg.add_define("USE_MIST_CORE")
