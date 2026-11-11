"""The v1.4.0 pack's plug-ins: what its id rules need that data cannot state.

The pack loader reads `PLUGINS`. Each name is the one a rule in `conventions.py` cites.

The three functions are still defined in `backend/knowledge/extraction/id_rails.py`. The
pack's rails hash covers the source of the module that defines a plug-in, so moving a
function's body into this file moves that hash: the move waits for the commit that
rewrites the release line.
"""

from __future__ import annotations

from backend.knowledge.extraction import id_rails
from backend.knowledge.ontologies.pack import ID, Plugin

TECHNOLOGY_VERSION = "technology-version"
METRIC_ID = "metric-id"
EVENT_DATE = "event-date"

PLUGINS: tuple[Plugin, ...] = (
    # A Technology's version, read from its name, leaves the end of its id.
    Plugin(TECHNOLOGY_VERSION, ID, id_rails.technology_version_id),
    # A Metric's id puts the value first, spelled as the metric rail spells it.
    Plugin(METRIC_ID, ID, id_rails.metric_id),
    # An Event dated to one ISO day gains that day at the end of its id.
    Plugin(EVENT_DATE, ID, id_rails.event_date_id),
)
