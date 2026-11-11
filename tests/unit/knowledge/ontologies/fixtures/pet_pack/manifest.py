"""The pet fixture pack's manifest: the base pack's next version."""

import dataclasses

from tests.unit.knowledge.ontologies.fixtures.base_pack.manifest import MANIFEST as BASE

MANIFEST = dataclasses.replace(BASE, version="0.2.0", parent_version=BASE.version)
