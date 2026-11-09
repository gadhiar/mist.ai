"""Binding an unbound epoch or cutover candidate to its full identity.

An epoch's full identity is its ledger triple plus a serving identity (the
extraction service's) and a dispatch identity (the backend's). Every epoch is
unbound when this code first runs against an existing database, and so is
every cutover open then. The dispatcher binds each one at its first successful
`/v1/info` contact, with `decide_binding`, which is pure: it reads what the
store and the cache already say and returns either a binding or a refusal.

THE NAMESPACE RULE
------------------
The cache rows of an (extraction_version, model_hash) pair have a 3-part
namespace (the key used before bindings existed) and any number of keyed ones
(`f"{serving_identity}+{dispatch_identity}"`). The 3-part namespace belongs to
the identity that first binds to it, its claim (`EventStore.legacy_claims`). A
target takes a keyed namespace only when it cannot share the 3-part rows, and
the active epoch never does: an upgrade must neither hide its rows nor force a
refill. Where a candidate would go keyed, the active epoch stalls on a
contradiction instead.

The rules, first match wins:

1. No change: a candidate whose full identity equals its bound source epoch's
   is refused. It is a settings-change candidate against a service that has
   not changed.
2. Operator: an acceptance row (`admin epoch accept`) whose identities equal
   the live ones binds in the 3-part namespace. Not for a candidate whose
   triple equals its source's: the 3-part rows of its pair are the source's.
3. Claimed by this identity: binds in the 3-part namespace.
4. Claimed by another identity: a candidate goes keyed; the active epoch
   stalls on the contradiction.
5. No claim:
   a. a candidate whose triple equals its source's goes keyed;
   b. no row of any outcome: binds in the 3-part namespace ('first-contact').
      A skip or dead-letter row counts, so its apply marker is never orphaned;
   c. otherwise the newest extracted row with a service record is compared
      with the service on adapter, llama_cpp_build and location_label.
      Agreement, or no such row, binds in the 3-part namespace ('adopted'). A
      difference sends a candidate keyed and stalls the active epoch.

THE EVIDENCE
------------
Each binding records what it verified and lists the rest as unverified. Only
the newest row is compared, which proves nothing about older rows; a matching
run identity verifies the settings the run identity hashes; context size,
placement, the device, top_p and the server flags are never recorded by any
row, so an adoption always lists them unverified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.extraction_contract.models import InfoResponse
from backend.knowledge.extraction_cache import AdoptionRecords, keyed_namespace

from .dispatch_identity import DispatchIdentity
from .store import Epoch

RULE_NO_CHANGE = "1"
RULE_OPERATOR = "2"
RULE_CLAIMED = "3"
RULE_CLAIMED_BY_OTHER = "4"
RULE_SETTINGS_CHANGE = "5a"
RULE_NO_ROWS = "5b"
RULE_RECORDS = "5c"

SOURCE_ADOPTED = "adopted"
SOURCE_FIRST_CONTACT = "first-contact"
SOURCE_OPERATOR = "operator"

# The fields a cache row's service record carries that rule 5c compares with
# the service. extraction_version and model_hash are not here: the service
# already passed the triple check before a binding is decided.
RECORDED_FIELDS: tuple[str, ...] = ("adapter", "llama_cpp_build", "location_label")

# The request-profile fields a matching run identity verifies: the run
# identity hashes the temperature, the constrained mode, the reasoning effort
# and budget, and the adapter (which decides whether thinking is sent at all).
# top_p is not among them: no run identity records it.
_RUN_IDENTITY_REQUEST_FIELDS: tuple[str, ...] = (
    "temperature",
    "constrained",
    "thinking_effort",
    "thinking_enabled",
    "thinking_budget_tokens",
)

# Dispatch components are named with this prefix in the evidence and the log,
# so a name says which identity it belongs to.
DISPATCH_NAME_PREFIX = "dispatch."


@dataclass(frozen=True, slots=True)
class Bind:
    """Bind the target.

    Attributes:
        rule: The Migration rule that matched (`RULE_*`).
        source: The binding source written to the store (`SOURCE_*`).
        identity_namespace: None for the 3-part namespace, else the keyed one.
        evidence: What the binding verified and could not, stored as JSON.
        unverified: The component names the binding could not verify, sorted.
    """

    rule: str
    source: str
    identity_namespace: str | None
    evidence: dict[str, Any]
    unverified: tuple[str, ...]

    @property
    def keyed(self) -> bool:
        """True when the target takes a keyed namespace."""
        return self.identity_namespace is not None


@dataclass(frozen=True, slots=True)
class Refuse:
    """Do not bind the target; the dispatcher stalls in `epoch_mismatch` with `reason`."""

    rule: str
    reason: str


def decide_binding(
    target: Epoch,
    info: InfoResponse,
    dispatch: DispatchIdentity,
    records: AdoptionRecords,
    claims: Sequence[tuple[str, str]],
    acceptance: Mapping[str, Any] | None,
    source: Epoch | None,
) -> Bind | Refuse:
    """Apply the binding rules (module docstring) to one unbound target.

    Args:
        target: The active epoch, or a candidate (`Cutover.target`).
        info: The service's `/v1/info`, which already passed the triple check
            against `target` and reports a serving identity.
        dispatch: The backend's live dispatch identity.
        records: `ExtractionCache.adoption_records` for the target's pair.
        claims: `EventStore.legacy_claims` for the target's pair.
        acceptance: The target's `epoch_identity_acceptance` row, or None.
        source: A candidate's source epoch with its binding; None for the
            active epoch or a candidate without one.

    Returns:
        A `Bind` or a `Refuse`.

    Raises:
        ValueError: `info` reports no serving identity; the dispatcher stalls
            before it gets here.
    """
    serving_identity = info.serving_identity
    if serving_identity is None:
        raise ValueError("a binding needs the service's serving identity")
    live = (serving_identity, dispatch.identity)
    candidate = target.cutover_id is not None
    same_triple = source is not None and _triple(source) == _triple(target)
    names = _component_names(info, dispatch)
    contradiction = _contradiction(target, info, records, claims, live)
    base = _base_evidence(info, records, claims)

    # Rule 1: a settings-change candidate against an unchanged service.
    if (
        candidate
        and same_triple
        and source is not None
        and (source.serving_identity, source.dispatch_identity) == live
    ):
        return Refuse(
            rule=RULE_NO_CHANGE,
            reason=(
                f"cutover {target.cutover_id}: the service still reports the active epoch's "
                "identity; restart it with the candidate settings"
            ),
        )

    # Rule 2: the operator accepted exactly the live identities.
    if (
        acceptance is not None
        and not (candidate and same_triple)
        and (acceptance.get("serving_identity"), acceptance.get("dispatch_identity")) == live
    ):
        verified = _verified_by_records(info, records)
        evidence = {
            **base,
            "rule": RULE_OPERATOR,
            "contradiction": contradiction,
            "acceptance": {
                "note": acceptance.get("note"),
                "accepted_at": acceptance.get("accepted_at"),
            },
        }
        return _bind(RULE_OPERATOR, SOURCE_OPERATOR, None, evidence, names, verified)

    # Rule 3: the pair's 3-part rows are claimed by this identity.
    if live in claims:
        evidence = {**base, "rule": RULE_CLAIMED}
        return _bind(RULE_CLAIMED, SOURCE_ADOPTED, None, evidence, names, set(names))

    keyed = keyed_namespace(*live)

    # Rule 4: claimed by another identity.
    if claims:
        if candidate:
            evidence = {**base, "rule": RULE_CLAIMED_BY_OTHER, "contradiction": contradiction}
            return _bind(RULE_CLAIMED_BY_OTHER, SOURCE_FIRST_CONTACT, keyed, evidence, names, set())
        return _refuse_active(RULE_CLAIMED_BY_OTHER, target, contradiction, live)

    # Rule 5a: a settings-change candidate whose source is unbound.
    if candidate and same_triple:
        evidence = {**base, "rule": RULE_SETTINGS_CHANGE}
        return _bind(RULE_SETTINGS_CHANGE, SOURCE_FIRST_CONTACT, keyed, evidence, names, set())

    # Rule 5b: nothing was ever cached for the pair.
    if records.rows == 0:
        evidence = {**base, "rule": RULE_NO_ROWS}
        return _bind(RULE_NO_ROWS, SOURCE_FIRST_CONTACT, None, evidence, names, set())

    # Rule 5c: compare the newest service-written row with the service.
    if contradiction is None:
        evidence = {**base, "rule": RULE_RECORDS}
        verified = _verified_by_records(info, records)
        return _bind(RULE_RECORDS, SOURCE_ADOPTED, None, evidence, names, verified)
    if candidate:
        evidence = {**base, "rule": RULE_RECORDS, "contradiction": contradiction}
        return _bind(RULE_RECORDS, SOURCE_FIRST_CONTACT, keyed, evidence, names, set())
    return _refuse_active(RULE_RECORDS, target, contradiction, live)


def binding_is_warning(bind: Bind) -> bool:
    """Whether the binding's log line is a WARNING (else INFO).

    An adopted or operator binding trusts settings it could not verify, and a
    candidate sent keyed by rule 4 or 5c will refill rows that already exist.
    """
    if bind.source in (SOURCE_ADOPTED, SOURCE_OPERATOR):
        return True
    return bind.keyed and bind.rule in (RULE_CLAIMED_BY_OTHER, RULE_RECORDS)


def binding_log_line(
    target: Epoch, bind: Bind, serving_identity: str, dispatch_identity: str
) -> str:
    """The one line every binding logs."""
    name = (
        f"cutover:{target.cutover_id}"
        if target.cutover_id is not None
        else f"epoch:{target.epoch_id}"
    )
    return (
        f"epoch_identity bound target={name} source={bind.source} rule={bind.rule} "
        f"serving_identity={serving_identity} dispatch_identity={dispatch_identity} "
        f"namespace={'keyed' if bind.keyed else '3-part'} "
        f"unverified={','.join(bind.unverified) or '-'}"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _triple(epoch: Epoch) -> tuple[str, str, str]:
    return (epoch.ontology_version, epoch.extraction_version, epoch.model_hash)


def _component_names(info: InfoResponse, dispatch: DispatchIdentity) -> tuple[str, ...]:
    serving = sorted((info.serving_identity_components or {}).keys())
    return (*serving, *(DISPATCH_NAME_PREFIX + n for n in sorted(dispatch.components)))


def _recorded(records: AdoptionRecords) -> dict[str, Any] | None:
    """The newest service-written row's values of `RECORDED_FIELDS`, or None."""
    if records.newest is None:
        return None
    stamps = records.newest.service_stamps
    result_stamps = stamps.get("stamps") or {}
    return {
        "adapter": result_stamps.get("adapter"),
        "llama_cpp_build": result_stamps.get("llama_cpp_build"),
        "location_label": stamps.get("location_label"),
    }


def _service_values(info: InfoResponse) -> dict[str, Any]:
    return {name: getattr(info, name) for name in RECORDED_FIELDS}


def _newest_run_identity(records: AdoptionRecords) -> str | None:
    if records.newest is None:
        return None
    return (records.newest.service_stamps.get("stamps") or {}).get("run_identity")


def _run_identity_verified(info: InfoResponse, records: AdoptionRecords) -> bool:
    newest = _newest_run_identity(records)
    return newest is not None and newest == info.run_identity


def _contradiction(
    target: Epoch,
    info: InfoResponse,
    records: AdoptionRecords,
    claims: Sequence[tuple[str, str]],
    live: tuple[str, str],
) -> str | None:
    """What the store says against binding `target` in the 3-part namespace, or None."""
    if claims and live not in claims:
        owners = " and ".join(f"{si}/{di}" for si, di in claims)
        return (
            f"the 3-part rows of {target.extraction_version}/{target.model_hash} "
            f"belong to {owners}"
        )
    if claims:
        return None
    recorded = _recorded(records)
    if recorded is None:
        return None
    service = _service_values(info)
    differing = [
        f"{name}={recorded[name]!r} where the service reports {service[name]!r}"
        for name in RECORDED_FIELDS
        if recorded[name] != service[name]
    ]
    if not differing:
        return None
    assert records.newest is not None  # _recorded returned values
    return f"the newest extracted row (event {records.newest.event_id}) records " + ", ".join(
        differing
    )


def _verified_by_records(info: InfoResponse, records: AdoptionRecords) -> set[str]:
    """The component names the pair's rows show to agree with the service."""
    recorded = _recorded(records)
    if recorded is None:
        return set()
    serving = info.serving_identity_components or {}
    service = _service_values(info)
    verified = {
        name for name in ("adapter", "llama_cpp_build") if recorded[name] == service[name]
    } & set(serving)
    if _run_identity_verified(info, records):
        shared = set(info.run_identity_components or {}) & set(serving)
        verified |= shared
        verified |= {
            name
            for name in serving
            if name.startswith("request.")
            and name.rsplit(".", 1)[-1] in _RUN_IDENTITY_REQUEST_FIELDS
        }
    return verified


def _base_evidence(
    info: InfoResponse, records: AdoptionRecords, claims: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    recorded = _recorded(records)
    return {
        "rows": records.rows,
        "extracted": records.extracted,
        "extracted_without_service_stamps": records.extracted_without_service_stamps,
        "run_identities": list(records.run_identities),
        "run_identity_verified": _run_identity_verified(info, records),
        "location_labels": list(records.location_labels),
        "newest_event_id": None if records.newest is None else records.newest.event_id,
        "recorded": (
            None
            if recorded is None
            else {
                name: {"row": recorded[name], "service": getattr(info, name)}
                for name in RECORDED_FIELDS
            }
        ),
        "claims": [list(claim) for claim in claims],
    }


def _bind(
    rule: str,
    source: str,
    namespace: str | None,
    evidence: dict[str, Any],
    names: Sequence[str],
    verified: set[str],
) -> Bind:
    unverified = tuple(name for name in names if name not in verified)
    evidence = {
        **evidence,
        "verified": sorted(name for name in names if name in verified),
        "unverified": list(unverified),
    }
    return Bind(
        rule=rule,
        source=source,
        identity_namespace=namespace,
        evidence=evidence,
        unverified=unverified,
    )


def _refuse_active(
    rule: str, target: Epoch, contradiction: str | None, live: tuple[str, str]
) -> Refuse:
    serving_identity, dispatch_identity = live
    return Refuse(
        rule=rule,
        reason=(
            f"epoch {target.epoch_id} cannot be adopted: {contradiction}; restore the service "
            "settings, accept them with admin epoch accept "
            f"--serving-identity {serving_identity} --dispatch-identity {dispatch_identity}, "
            "or begin a cutover with --settings-change --refill"
        ),
    )
