"""The gates are not gates until the runnable command calls them. MIS-137.

## The gap these tests close

Every non-vacuity floor in `rebuild_gate.py` was called only from tests, and the
one comparison the CLI ran was a `print`:

    print(live_vs_rebuilt_report(live_form, build_b))   # mist_admin.py:1125
    return 0                                            # mist_admin.py:1126

So `mist_admin graph-rebuild-from-log` exited 0 on a live-vs-rebuilt divergence of
any size, and `hydrate && rebuild && echo GREEN` printed GREEN on a red result.
`rebuild_gate.live_vs_rebuilt_report`'s own docstring said so -- *"Diagnostic (NOT
a gate in R1.2)"* -- which made it an accurate function and an inaccurate gate.

Two shapes of assertion, deliberately, following `test_rebuild_scoping.py`:

- **The call.** `TestTheCommandWiresTheGates` fails if the command stops invoking a
  gate, independent of whether any particular graph would trip it. This is the
  assertion whose absence let five fully-tested, mutation-proven floors sit unused.
- **The consequence.** `TestExitCodes` proves each gate maps to a DISTINCT exit
  code, so an operator can tell non-determinism from a derivation gap from a run
  that proved nothing. One shared failure code would make the gates indistinguishable
  to the only consumer that reads them.

## Why a separate exception type for live-vs-rebuilt

`assert_rebuild_twice_identical` failing means **non-determinism**: the same inputs
produced two different graphs. `assert_live_equals_rebuilt` failing means a
**derivation gap**: the rebuild is deterministic and disagrees with live. Different
causes, different remedies, and a caller collapsing both into one code cannot say
which happened. Hence `RebuildDivergenceError`, distinct from
`RebuildDeterminismError`, asserted here so a future refactor cannot quietly merge
them.

## Why the gate parameters are required rather than defaulted

`assert_turns_processed` and `assert_replay_derived_non_vacuous` both refuse a
floor below 1 with `ValueError`, because a floor of zero is satisfied by exactly
the emptiness they exist to refuse. The same reasoning applies one level up: a
DEFAULTED floor is a floor nobody chose. So the command refuses to run without
`--expect-turns` and `--min-replay-edges`, and refuses BEFORE connecting -- the
same fail-closed ordering `test_cmd_hydrate.py` pins for the isolation check.

`--diagnostic` is the explicit escape, and it prints a banner saying no gate ran.
That is deliberately different from a safety guard's "off" switch, which this repo
refuses to add (`eval_isolation.py:593-596`). This is a measurement mode: the
hazard is a silent pass, and an escape that announces itself is not silent.
"""

from __future__ import annotations

import argparse
import json

import pytest

from backend.knowledge.regeneration.rebuild_gate import (
    RebuildDeterminismError,
    RebuildDivergenceError,
    RebuildVacuityError,
    assert_live_equals_rebuilt,
)
from scripts import mist_admin

LIVE_URI = "bolt://mist-neo4j:7687"
STAGING_URI = "bolt://mist-neo4j-staging:7687"


def _form(
    *,
    nodes: int = 2,
    replay_edges: int = 2,
    tag: str = "a",
    self_model_nodes: int = 21,
) -> str:
    """A canonical-form string with `replay_edges` edges carrying both markers.

    `source_utterance_id` and `version_key` together are `REPLAY_EDGE_MARKERS` --
    the pair only a replay writes, which is what separates the replay floor from
    the whole-graph node floor.
    """
    payload = {
        "nodes": [
            {"id": f"n{i}", "labels": ["__Entity__"], "properties": {}} for i in range(nodes)
        ],
        "relationships": [
            {
                "source": f"n{i}",
                "type": "USES",
                "target": f"n{i + 1}",
                "properties": {
                    "source_utterance_id": f"e{i}",
                    "version_key": f"e{i}|t|-",
                    "tag": tag,
                },
            }
            for i in range(replay_edges)
        ],
        # MIS-130 step C put the self-model partition INTO the compared surface,
        # so a form the gate reads now carries it. `self_model_nodes=0` is how a
        # test drives the case `assert_self_model_applied` exists to refuse: a
        # rebuild that applied nothing, which every OTHER gate certifies as
        # correct because two empty partitions are byte-identical.
        "self_model": {
            "nodes": [
                {"id": f"sm{i}", "labels": ["__SelfModel__"], "properties": {}}
                for i in range(self_model_nodes)
            ],
            "relationships": [],
        },
        # Present and empty so the fake form carries the shape the extended surface
        # actually produces. No parameter to vary them: the tests that need a
        # populated cross-layer key assert at the FORM level, in
        # test_self_model_comparison_surface.py and
        # test_self_model_provenance_surface.py, where the switch genuinely decides
        # whether the key exists. This fixture's `canonical_graph_form` fake ignores
        # the switches, so a dropped-edge test written here would pass with the
        # surface turned off -- proving only that the equality gate fires on forms
        # that differ, which was never in question. Two such tests were written and
        # deleted for exactly that reason.
        "self_model_cross_layer_edges": [],
        "self_model_provenance_edges": [],
    }
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


class TestTheAssertionExists:
    """`live == rebuilt` must be an assertion, not a string."""

    def test_equal_forms_pass(self):
        """Not raising IS the assertion; there is no return value to check."""
        form = _form()
        assert_live_equals_rebuilt(form, form)

    def test_divergent_forms_raise(self):
        with pytest.raises(RebuildDivergenceError) as exc:
            assert_live_equals_rebuilt(_form(tag="live"), _form(tag="rebuilt"))
        assert "live != rebuilt" in str(exc.value)

    def test_the_message_carries_the_diff(self):
        """A gate that fails without saying where is a gate someone disables."""
        with pytest.raises(RebuildDivergenceError) as exc:
            assert_live_equals_rebuilt(_form(tag="live"), _form(tag="rebuilt"))
        body = str(exc.value)
        assert "--- live" in body and "+++ rebuilt" in body
        assert "live" in body and "rebuilt" in body

    def test_divergence_is_not_a_determinism_error(self):
        """The two failures have different causes and must stay distinguishable."""
        assert not issubclass(RebuildDivergenceError, RebuildDeterminismError)
        assert not issubclass(RebuildDeterminismError, RebuildDivergenceError)


class _FakeConn:
    def __init__(self):
        self.writes = []
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def execute_write(self, query, params=None):
        self.writes.append((query, params))
        return []


class _FakeReport:
    def __init__(self, turns_processed: int):
        self.job_id = "job-1"
        self.turns_processed = turns_processed
        self.turns_failed = 0
        self.staging_uri = STAGING_URI
        self.epoch_id = 1
        # The domain the selection ran under (MIS-138). Present here because the
        # command prints it from the REPORT rather than from its own args, so a
        # report missing these fields is a real breakage rather than a fake's gap.
        self.ontology_version = "1.4.0"
        self.origins = ("real",)
        self.total_logged = turns_processed


class _FakeRegen:
    def __init__(self, turns_processed: int):
        self._turns = turns_processed
        self.rebuild_calls = 0

    async def rebuild(self, **kwargs):
        self.rebuild_calls += 1
        return _FakeReport(self._turns)


def _args(**kw) -> argparse.Namespace:
    base = {
        "dry_run": True,
        "staging_uri": STAGING_URI,
        "epoch": None,
        "expect_turns": 87,
        "min_replay_edges": 2,
        # MIS-130 step B: the third required floor. Sized from the seed corpus
        # (`mist-memory/seed/mist.md` authors 21 :__SelfModel__ nodes), same as the
        # other two are sized from the replay corpus.
        "min_seed_nodes": 21,
        "diagnostic": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def wired(monkeypatch):
    """Stub the command's collaborators and record what the gates saw.

    The command imports `canonical_graph_form` INSIDE the function body, so
    patching the source module intercepts it -- the import resolves the attribute
    at call time.
    """
    state = {"forms": [], "regen": None, "live_form": None, "rebuilt_forms": []}

    class _Cfg:
        class neo4j:  # noqa: N801 -- mirrors the real config's attribute shape
            uri = LIVE_URI
            username = "neo4j"
            password = "password"

    class _Backend:
        Neo4jConnection = staticmethod(lambda cfg: _FakeConn())

        @staticmethod
        def get_config():
            return _Cfg()

    monkeypatch.setattr(mist_admin, "_load_backend", lambda: _Backend())
    monkeypatch.setattr(mist_admin, "_connect", lambda be: _FakeConn())

    def _set(*, turns=87, live=None, rebuilt=None):
        """Install the regenerator and the forms each connection serialises to."""
        state["regen"] = _FakeRegen(turns)
        state["live_form"] = live if live is not None else _form(tag="same")
        state["rebuilt_forms"] = list(rebuilt or [_form(tag="same"), _form(tag="same")])

        monkeypatch.setattr(
            mist_admin,
            "_build_log_regenerator",
            lambda be, staging_conn, epoch_id: (state["regen"], {"epoch_id": 1}),
        )

        calls = {"n": 0}

        def _canon(connection, **kwargs):
            # The command serialises staging twice, then live once, in that order.
            state["forms"].append(kwargs)
            if calls["n"] < len(state["rebuilt_forms"]):
                form = state["rebuilt_forms"][calls["n"]]
                calls["n"] += 1
                return form
            return state["live_form"]

        monkeypatch.setattr("backend.knowledge.canonical_serialize.canonical_graph_form", _canon)

    state["set"] = _set
    return state


class TestFailsClosedWithoutChosenFloors:
    """A defaulted floor is a floor nobody chose."""

    def test_missing_expect_turns_refuses_before_connecting(self, wired, monkeypatch, capsys):
        connected = []
        monkeypatch.setattr(mist_admin, "_connect", lambda be: connected.append(1))
        wired["set"]()

        assert mist_admin.cmd_graph_rebuild_from_log(_args(expect_turns=None)) == 2

        assert connected == [], "refused, but only after opening a live connection"
        assert "--expect-turns" in capsys.readouterr().out

    def test_missing_min_replay_edges_refuses(self, wired, capsys):
        wired["set"]()
        assert mist_admin.cmd_graph_rebuild_from_log(_args(min_replay_edges=None)) == 2
        assert "--min-replay-edges" in capsys.readouterr().out

    def test_missing_min_seed_nodes_refuses(self, wired, capsys):
        """The seed floor is required on the same terms as the other two.

        MIS-130 step B. Without it a seed-apply that writes zero nodes produces an
        empty `:__SelfModel__` partition, and every other gate certifies it: two
        empty partitions are byte-identical, so determinism passes, and the
        compared surface does not read that partition, so equality passes too.
        """
        wired["set"]()
        assert mist_admin.cmd_graph_rebuild_from_log(_args(min_seed_nodes=None)) == 2
        assert "--min-seed-nodes" in capsys.readouterr().out

    def test_no_rebuild_is_attempted_when_refused(self, wired):
        wired["set"]()
        mist_admin.cmd_graph_rebuild_from_log(_args(expect_turns=None))
        assert wired["regen"].rebuild_calls == 0


class TestTheCommandWiresTheGates:
    """The call assertions -- these fail if a gate stops being invoked."""

    def test_all_gates_green_returns_zero(self, wired):
        wired["set"]()
        assert mist_admin.cmd_graph_rebuild_from_log(_args()) == 0

    def test_live_form_is_serialised_with_the_same_switches_as_staging(self, wired):
        """A comparison across different switch sets compares different surfaces."""
        wired["set"]()
        mist_admin.cmd_graph_rebuild_from_log(_args())

        assert len(wired["forms"]) == 3, "expected two staging builds and one live"
        assert wired["forms"][0] == wired["forms"][1] == wired["forms"][2]

    def test_every_form_includes_the_self_model_partition(self, wired):
        """MIS-130 step C. Until now the gate compared `:__Entity__` only.

        On the live graph that was 11 nodes of 32 -- twenty-one structurally
        invisible. Safe to turn on only now that step A retired the copy-forward:
        while the copy stood, this compared the partition against a copy of the
        comparison's own left-hand side, which is green by construction and would
        have been presented as new coverage.
        """
        wired["set"]()
        mist_admin.cmd_graph_rebuild_from_log(_args())

        assert wired["forms"], "no form was serialised at all"
        assert all(k.get("include_self_model") for k in wired["forms"]), (
            f"a form was built without include_self_model: {wired['forms']}. The "
            "gate would report the self-model verified while never having looked."
        )

    def test_every_form_includes_provenance_so_the_fourth_clause_is_reachable(self, wired):
        """MIS-139: self-model <-> provenance needs BOTH switches to be emitted."""
        wired["set"]()
        mist_admin.cmd_graph_rebuild_from_log(_args())

        assert all(k.get("include_provenance") for k in wired["forms"]), (
            "provenance is off, so `self_model_provenance_edges` is absent from "
            "every form and a dropped LEARNED_SELF edge stays invisible."
        )

    def test_self_model_gate_is_wired(self, wired, capsys):
        """A rebuild that applied no self-model must fail, not pass."""
        wired["set"](
            live=_form(tag="same", self_model_nodes=21),
            rebuilt=[
                _form(tag="same", self_model_nodes=0),
                _form(tag="same", self_model_nodes=0),
            ],
        )
        assert mist_admin.cmd_graph_rebuild_from_log(_args()) == 4
        assert "self-model gate FAILED" in capsys.readouterr().out

    def test_turns_gate_is_wired(self, wired, capsys):
        wired["set"](turns=86)
        assert mist_admin.cmd_graph_rebuild_from_log(_args(expect_turns=87)) == 4
        assert "turns gate FAILED" in capsys.readouterr().out

    def test_replay_floor_is_wired(self, wired, capsys):
        wired["set"](
            live=_form(replay_edges=1, tag="same"),
            rebuilt=[_form(replay_edges=1, tag="same")] * 2,
        )
        assert mist_admin.cmd_graph_rebuild_from_log(_args(min_replay_edges=2)) == 4
        assert "replay non-vacuity gate FAILED" in capsys.readouterr().out

    def test_node_floor_is_wired(self, wired, capsys):
        empty = json.dumps({"nodes": [], "relationships": []}, indent=2) + "\n"
        wired["set"](live=empty, rebuilt=[empty, empty])
        assert mist_admin.cmd_graph_rebuild_from_log(_args()) == 4
        assert "FAILED" in capsys.readouterr().out

    def test_live_equals_rebuilt_is_wired(self, wired, capsys):
        wired["set"](live=_form(tag="live"), rebuilt=[_form(tag="rebuilt")] * 2)
        assert mist_admin.cmd_graph_rebuild_from_log(_args()) == 3
        assert "live != rebuilt" in capsys.readouterr().out

    def test_rebuild_twice_gate_still_wired(self, wired, capsys):
        wired["set"](rebuilt=[_form(tag="one"), _form(tag="two")])
        assert mist_admin.cmd_graph_rebuild_from_log(_args()) == 1
        assert "determinism gate FAILED" in capsys.readouterr().out


class TestExitCodes:
    """Each failure mode gets its own code, so the operator learns which one it was."""

    def test_codes_are_distinct(self, wired, capsys):
        wired["set"](rebuilt=[_form(tag="one"), _form(tag="two")])
        non_determinism = mist_admin.cmd_graph_rebuild_from_log(_args())
        capsys.readouterr()

        wired["set"](live=_form(tag="live"), rebuilt=[_form(tag="rebuilt")] * 2)
        divergence = mist_admin.cmd_graph_rebuild_from_log(_args())
        capsys.readouterr()

        wired["set"](turns=1)
        vacuity = mist_admin.cmd_graph_rebuild_from_log(_args(expect_turns=87))
        capsys.readouterr()

        wired["set"]()
        green = mist_admin.cmd_graph_rebuild_from_log(_args())

        assert len({non_determinism, divergence, vacuity, green}) == 4, (
            "non-determinism, divergence, vacuity and success must be "
            f"distinguishable; got {non_determinism}, {divergence}, {vacuity}, {green}"
        )


class TestDiagnosticModeAnnouncesItself:
    """The escape is explicit and loud, which is what makes it not a silent pass."""

    def test_diagnostic_tolerates_divergence(self, wired):
        wired["set"](live=_form(tag="live"), rebuilt=[_form(tag="rebuilt")] * 2)
        assert mist_admin.cmd_graph_rebuild_from_log(_args(diagnostic=True)) == 0

    def test_diagnostic_says_no_gate_ran(self, wired, capsys):
        wired["set"](live=_form(tag="live"), rebuilt=[_form(tag="rebuilt")] * 2)
        mist_admin.cmd_graph_rebuild_from_log(_args(diagnostic=True))
        out = capsys.readouterr().out
        assert "DIAGNOSTIC" in out
        assert "not a gate" in out.lower()

    def test_diagnostic_needs_no_floors(self, wired):
        """Floors are what it is opting out of; requiring them would be theatre."""
        wired["set"]()
        assert (
            mist_admin.cmd_graph_rebuild_from_log(
                _args(diagnostic=True, expect_turns=None, min_replay_edges=None)
            )
            == 0
        )


def test_vacuity_error_is_not_a_divergence_error():
    """Three failure families, three types, so the CLI can map three exit codes."""
    assert not issubclass(RebuildVacuityError, RebuildDivergenceError)
    assert not issubclass(RebuildDivergenceError, RebuildVacuityError)
