"""MIS-140 T2: the two gates that decide WHERE a restore is allowed to write.

Everything here is a refusal test. The positive path is one line; the value of
this module is the list of things it will not accept.
"""

from __future__ import annotations

import pytest

from backend.knowledge.eval_isolation import REPO_ROOT
from scripts.backup.errors import RestoreConfirmationError, RestoreTargetError
from scripts.backup.target import (
    RESTORE_MARKER_FILENAME,
    assert_restore_target_root,
    assert_target_confirmed,
    restore_marker_path,
)


class TestTheHandshake:
    def test_a_marked_directory_is_accepted_and_returned_resolved(self, restore_target):
        resolved = assert_restore_target_root(restore_target)
        assert resolved == restore_target.resolve()
        assert resolved.is_absolute()

    def test_an_unmarked_directory_is_refused(self, tmp_path):
        unmarked = tmp_path / "somewhere"
        unmarked.mkdir()
        with pytest.raises(RestoreTargetError) as refusal:
            assert_restore_target_root(unmarked)
        # The refusal has to name the exact file to create, because the
        # operator reading it is not reading target.py.
        assert RESTORE_MARKER_FILENAME in str(refusal.value)
        assert str(restore_marker_path(unmarked)) in str(refusal.value)

    def test_the_marker_must_be_a_file_not_a_directory(self, tmp_path):
        root = tmp_path / "target"
        (root / RESTORE_MARKER_FILENAME).mkdir(parents=True)
        with pytest.raises(RestoreTargetError):
            assert_restore_target_root(root)

    def test_an_absent_target_is_refused_rather_than_created(self, tmp_path):
        absent = tmp_path / "not-there"
        with pytest.raises(RestoreTargetError):
            assert_restore_target_root(absent)
        assert not absent.exists()

    def test_a_symlink_to_an_unmarked_directory_is_refused(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("this platform does not allow creating symlinks unprivileged")
        with pytest.raises(RestoreTargetError):
            assert_restore_target_root(link)

    def test_the_marker_is_not_consumed_so_a_target_stays_marked(self, restore_target):
        assert_restore_target_root(restore_target)
        assert_restore_target_root(restore_target)
        assert restore_marker_path(restore_target).is_file()


class TestLiveStateIsRefusedBeforeTheMarkerIsConsidered:
    def test_the_live_state_root_is_refused(self):
        with pytest.raises(RestoreTargetError) as refusal:
            assert_restore_target_root(REPO_ROOT / "data")
        # Refused as LIVE, not as unmarked: an "unmarked" message would invite
        # the operator to fix it by creating the marker in live state.
        assert RESTORE_MARKER_FILENAME not in str(refusal.value)

    def test_a_directory_under_live_state_is_refused(self):
        with pytest.raises(RestoreTargetError):
            assert_restore_target_root(REPO_ROOT / "data" / "restore-here")

    def test_a_root_that_contains_live_state_is_refused(self):
        with pytest.raises(RestoreTargetError):
            assert_restore_target_root(REPO_ROOT)

    def test_the_dev_hydration_state_root_is_a_legitimate_target(self, tmp_path):
        # `<repo>/dev-state` is inside the repository, which makes it an illegal
        # backup DESTINATION and a perfectly good restore TARGET. The two
        # guards are therefore different guards, and this test is what says so:
        # `assert_backup_destination` refuses this path
        # (`tests/unit/backup/test_destination.py`), and this one accepts it.
        dev_state = tmp_path / "dev-state"
        dev_state.mkdir()
        (dev_state / RESTORE_MARKER_FILENAME).touch()
        assert assert_restore_target_root(dev_state) == dev_state.resolve()


class TestTheTypedConfirmation:
    def test_the_resolved_path_typed_back_is_accepted(self, restore_target):
        resolved = assert_restore_target_root(restore_target)
        assert assert_target_confirmed(resolved, str(resolved)) is None

    def test_surrounding_whitespace_is_stripped(self, restore_target):
        resolved = assert_restore_target_root(restore_target)
        assert assert_target_confirmed(resolved, f"  {resolved}\n") is None

    def test_no_token_is_refused_and_the_refusal_prints_the_string_to_type(self, restore_target):
        resolved = assert_restore_target_root(restore_target)
        with pytest.raises(RestoreConfirmationError) as refusal:
            assert_target_confirmed(resolved, None)
        assert str(resolved) in str(refusal.value)

    def test_an_empty_token_is_refused(self, restore_target):
        resolved = assert_restore_target_root(restore_target)
        with pytest.raises(RestoreConfirmationError):
            assert_target_confirmed(resolved, "   ")

    def test_a_different_path_is_refused(self, restore_target, tmp_path):
        resolved = assert_restore_target_root(restore_target)
        with pytest.raises(RestoreConfirmationError):
            assert_target_confirmed(resolved, str(tmp_path / "somewhere-else"))

    def test_retyping_a_relative_path_does_not_confirm_it(self, restore_target, monkeypatch):
        # The point of comparing against the RESOLVED path. Typing back the same
        # relative string the operator passed proves nothing about where it
        # lands, so it is refused even though the two strings are equal to each
        # other and the path is correct.
        monkeypatch.chdir(restore_target.parent)
        resolved = assert_restore_target_root("./dev-state")
        assert resolved == restore_target.resolve()
        with pytest.raises(RestoreConfirmationError):
            assert_target_confirmed(resolved, "./dev-state")

    def test_a_trailing_separator_is_refused(self, restore_target):
        # Not pedantry: normalising it would mean the compared string is no
        # longer the string the refusal printed, and every extra equivalence
        # widens what counts as confirmation.
        resolved = assert_restore_target_root(restore_target)
        with pytest.raises(RestoreConfirmationError):
            assert_target_confirmed(resolved, f"{resolved}/")
