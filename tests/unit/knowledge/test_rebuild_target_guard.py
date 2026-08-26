"""Unit tests for the rebuild target-isolation guard (R1.2, hardened 2026-08-04).

The guard is an ALLOWLIST of disposable staging endpoints. It was a denylist of
one spelling, and `TestTheHoleTheDenylistLeft` below is the case that was live:
`bolt://localhost:7687` is textually unlike `bolt://mist-neo4j:7687` and is the
same database, because the live bolt port is host-published. The rebuild
`DETACH DELETE`s its target, so that spelling wiped the canonical graph.

`test_rejects_host_published_live_port_alias` claimed to cover it and did not:
it passed `localhost:7687` as BOTH target and live, so it only ever exercised
the equality arm and would have passed against a guard that did nothing else.
It is rewritten below to pass the live URI the caller actually passes.
"""

import pytest

from backend.knowledge.eval_isolation import RebuildTargetError, assert_rebuild_target_not_live

LIVE = "bolt://mist-neo4j:7687"


class TestAllowsDisposableStagingEndpoints:
    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("bolt://mist-neo4j-staging:7687", id="staging-in-network"),
            pytest.param("bolt://localhost:7689", id="staging-host-published"),
            pytest.param("bolt://127.0.0.1:7689", id="staging-loopback"),
        ],
    )
    def test_allows_the_staging_instance(self, target):
        assert_rebuild_target_not_live(target_uri=target, live_uri=LIVE)


class TestTheHoleTheDenylistLeft:
    """Both of these PASSED the old guard and wiped the live graph."""

    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://127.0.0.1:7687", id="live-loopback"),
        ],
    )
    def test_refuses_a_host_side_alias_of_the_live_instance(self, target):
        # The live URI is the in-network service name, exactly as `mist_admin`
        # and `LogRegenerator` pass it -- so the target is textually unlike live
        # and addresses the same database. This is the whole finding.
        with pytest.raises(RebuildTargetError):
            assert_rebuild_target_not_live(target_uri=target, live_uri=LIVE)


class TestRefusesEverythingNotOnTheAllowlist:
    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://MIST-NEO4J:7687", id="live-uppercased"),
            pytest.param("neo4j://mist-neo4j:7687", id="live-neo4j-scheme"),
            pytest.param("bolt://mist-neo4j-eval:7687", id="eval-is-not-scratch-space"),
            pytest.param("bolt://localhost:7688", id="eval-host-published"),
            pytest.param("bolt://mist-neo4j-dev:7687", id="dev-holds-the-hydrated-fixture"),
            pytest.param("bolt://localhost:7690", id="dev-host-published"),
            pytest.param("bolt://some-unknown-host:7687", id="unknown-host"),
        ],
    )
    def test_refuses(self, target):
        with pytest.raises(RebuildTargetError):
            assert_rebuild_target_not_live(target_uri=target, live_uri=LIVE)

    def test_the_message_says_it_needs_to_be_on_the_list(self):
        with pytest.raises(RebuildTargetError, match="not a recognized disposable"):
            assert_rebuild_target_not_live(target_uri="bolt://elsewhere:7687", live_uri=LIVE)


class TestUnparseableInput:
    def test_rejects_unparseable_target(self):
        with pytest.raises(RebuildTargetError, match="parse"):
            assert_rebuild_target_not_live(target_uri="not-a-uri", live_uri=LIVE)

    def test_rejects_a_target_with_no_port(self):
        # Live and staging differ only by port on the host, so a portless target
        # cannot be distinguished from the live graph.
        with pytest.raises(RebuildTargetError, match="port"):
            assert_rebuild_target_not_live(target_uri="bolt://mist-neo4j-staging", live_uri=LIVE)

    def test_rejects_unparseable_live_uri(self):
        with pytest.raises(RebuildTargetError, match="parse"):
            assert_rebuild_target_not_live(
                target_uri="bolt://mist-neo4j-staging:7687", live_uri="not-a-uri"
            )


class TestAllowlistOverride:
    def test_override_admits_a_ci_endpoint(self, monkeypatch):
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "ci-neo4j:9999")
        assert_rebuild_target_not_live(target_uri="bolt://ci-neo4j:9999", live_uri=LIVE)
        with pytest.raises(RebuildTargetError):
            assert_rebuild_target_not_live(
                target_uri="bolt://mist-neo4j-staging:7687", live_uri=LIVE
            )

    def test_widening_the_override_to_include_live_still_refuses(self, monkeypatch):
        # An operator cannot hand themselves the wipe by adding the live
        # endpoint to their own allowlist.
        #
        # This test used to match "resolves to the live graph", pinning the
        # `live_uri` equality arm specifically. The hardcoded live denylist now
        # fires FIRST for this spelling, so the message changed while the
        # refusal did not. Matching on "live" keeps the test's actual subject --
        # that the override cannot admit live -- without pinning WHICH arm
        # catches it. The arm-specific case is
        # `test_the_live_uri_arm_still_catches_a_non_denylisted_live_endpoint`.
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "mist-neo4j:7687")
        with pytest.raises(RebuildTargetError, match="live"):
            assert_rebuild_target_not_live(target_uri=LIVE, live_uri=LIVE)

    def test_the_live_uri_arm_still_catches_a_non_denylisted_live_endpoint(self, monkeypatch):
        """The `live_uri` arm is not made redundant by the denylist.

        The denylist covers the three canonical spellings of THIS deployment's
        live graph. A live instance reachable at some other address -- a remote
        host, a relocated port -- is outside it, and only the `live_uri`
        comparison catches that. Both arms are load-bearing, for different
        threats: the denylist against an operator's override, this against a
        deployment the denylist does not know about.
        """
        remote_live = "bolt://neo4j.internal.example:7687"
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "neo4j.internal.example:7687")

        with pytest.raises(RebuildTargetError, match="resolves to the live graph"):
            assert_rebuild_target_not_live(target_uri=remote_live, live_uri=remote_live)

    def test_empty_override_refuses_rather_than_allowing_everything(self, monkeypatch):
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", " , ")
        with pytest.raises(RebuildTargetError, match="empty allowlist"):
            assert_rebuild_target_not_live(
                target_uri="bolt://mist-neo4j-staging:7687", live_uri=LIVE
            )

    def test_malformed_override_raises_the_type_the_cli_catches(self, monkeypatch):
        # `mist_admin` catches RebuildTargetError to print a clean refusal; an
        # EvalIsolationError leaking through would read as a crash instead.
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "no-port-here")
        with pytest.raises(RebuildTargetError, match="Malformed MIST_REBUILD_NEO4J_HOSTS"):
            assert_rebuild_target_not_live(
                target_uri="bolt://mist-neo4j-staging:7687", live_uri=LIVE
            )

    @pytest.mark.parametrize(
        "live_uri",
        [
            pytest.param("bolt://mist-neo4j-dev:7687", id="live-uri-inferred-as-dev"),
            pytest.param("bolt://mist-neo4j-staging:7687", id="live-uri-inferred-as-staging"),
        ],
    )
    def test_refuses_a_live_target_even_when_live_uri_is_not_live(self, monkeypatch, live_uri):
        """The second arm is only as good as the `live_uri` it is handed.

        `cmd_graph_rebuild_from_log` INFERS live_uri from ambient config
        (`be.get_config().neo4j.uri`). The rebuild can only run inside
        `mist-backend-dev` -- the sole process whose EVENT_STORE_DB_PATH points
        at the hydrated store -- and there `NEO4J_URI` is the DEV instance. So
        in the only place the command can run, the "target resolves to live" arm
        compares the target against the dev graph and is vacuous.

        Combine that with a widened `MIST_REBUILD_NEO4J_HOSTS` (which REPLACES
        the allowlist) and both arms pass on a live target. The hardcoded live
        denylist is the arm no caller and no environment can weaken.
        """
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "mist-neo4j:7687,localhost:7687")

        with pytest.raises(RebuildTargetError, match="live"):
            assert_rebuild_target_not_live(target_uri=LIVE, live_uri=live_uri)

    def test_refuses_the_host_published_live_spelling_under_a_widened_override(self, monkeypatch):
        """`bolt://localhost:7687` is the live graph -- the port is host-published.

        This is the exact spelling that caused the original incident the
        allowlist was introduced to prevent.
        """
        monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "localhost:7687")

        with pytest.raises(RebuildTargetError, match="live"):
            assert_rebuild_target_not_live(
                target_uri="bolt://localhost:7687",
                live_uri="bolt://mist-neo4j-dev:7687",
            )

    def test_staging_still_passes_with_the_denylist_in_place(self):
        """Pairing guard: the denylist must not refuse the one target this permits."""
        assert_rebuild_target_not_live(
            target_uri="bolt://mist-neo4j-staging:7687",
            live_uri=LIVE,
        )
