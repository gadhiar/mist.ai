"""Tests for backend.user_timezone.resolve_user_timezone, the one reader of `MIST_TIMEZONE`.

The valid-name cases use the real zone database (tzdata is in the image). The missing-database
case patches both the zone lookup and `available_timezones`, so it runs whatever is installed.
"""

import logging
import zoneinfo
from datetime import UTC, timedelta, timezone

import pytest

import backend.user_timezone as user_timezone
from backend.user_timezone import USER_TIMEZONE_ENV, resolve_user_timezone

_LOGGER = user_timezone.__name__


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    """Each test sees a fresh process: no warning has been logged yet."""
    monkeypatch.setattr(user_timezone, "_unset_warned", False)
    monkeypatch.setattr(user_timezone, "_missing_database_warned", set())


def _not_found(name: str):
    raise zoneinfo.ZoneInfoNotFoundError(f"No time zone found with key {name}")


def test_the_variable_is_mist_timezone():
    assert USER_TIMEZONE_ENV == "MIST_TIMEZONE"


@pytest.mark.parametrize("environ", [{}, {"MIST_TIMEZONE": ""}, {"MIST_TIMEZONE": "   "}])
def test_unset_or_blank_is_utc(environ):
    zone = resolve_user_timezone(environ)

    assert zone is UTC


def test_unset_warns_once_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        zones = [resolve_user_timezone({}) for _ in range(3)]

    warnings = [r for r in caplog.records if "MIST_TIMEZONE" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert all(zone is UTC for zone in zones)


def test_a_valid_name_gives_its_zone():
    zone = resolve_user_timezone({"MIST_TIMEZONE": "Asia/Tokyo"})

    assert isinstance(zone, zoneinfo.ZoneInfo)
    assert zone.key == "Asia/Tokyo"


def test_a_valid_name_is_stripped():
    zone = resolve_user_timezone({"MIST_TIMEZONE": "  America/Chicago "})

    assert zone == zoneinfo.ZoneInfo("America/Chicago")


def test_a_valid_name_never_lists_the_database(monkeypatch):
    """The zone is tried first, so a usage request never walks the zone database."""

    def listing_forbidden():
        raise AssertionError("available_timezones must not be called for a valid zone")

    monkeypatch.setattr(user_timezone, "available_timezones", listing_forbidden)

    assert resolve_user_timezone({"MIST_TIMEZONE": "Europe/London"}).key == "Europe/London"


def test_a_valid_name_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        resolve_user_timezone({"MIST_TIMEZONE": "Europe/London"})

    assert caplog.records == []


def test_a_missing_database_gives_utc_with_one_warning_naming_the_zone(monkeypatch, caplog):
    monkeypatch.setattr(user_timezone, "ZoneInfo", _not_found)
    monkeypatch.setattr(user_timezone, "available_timezones", set)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        zones = [resolve_user_timezone({"MIST_TIMEZONE": "Asia/Tokyo"}) for _ in range(2)]

    assert all(zone is UTC for zone in zones)
    warnings = [r for r in caplog.records if "Asia/Tokyo" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_an_unknown_name_with_the_database_present_raises_value_error(monkeypatch):
    monkeypatch.setattr(user_timezone, "ZoneInfo", _not_found)
    monkeypatch.setattr(user_timezone, "available_timezones", lambda: {"Asia/Tokyo"})

    with pytest.raises(ValueError, match="Mars/Olympus_Mons") as excinfo:
        resolve_user_timezone({"MIST_TIMEZONE": "Mars/Olympus_Mons"})

    assert isinstance(excinfo.value.__cause__, zoneinfo.ZoneInfoNotFoundError)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("Mars/Olympus_Mons", id="unknown"),
        pytest.param("America", id="a-directory-of-zones"),
        pytest.param("/etc/localtime", id="absolute-path"),
        pytest.param("../etc/passwd", id="outside-the-database"),
    ],
)
def test_an_invalid_name_raises_value_error_naming_the_variable(name):
    with pytest.raises(ValueError, match="MIST_TIMEZONE"):
        resolve_user_timezone({"MIST_TIMEZONE": name})


def test_an_invalid_name_is_not_a_key_error():
    """`ZoneInfoNotFoundError` is a KeyError; callers catch ValueError only."""
    with pytest.raises(ValueError) as excinfo:
        resolve_user_timezone({"MIST_TIMEZONE": "Mars/Olympus_Mons"})

    assert not isinstance(excinfo.value, KeyError)


def test_an_injected_loader_replaces_the_lookup():
    zone = timezone(timedelta(hours=3), "TEST+3")
    names: list[str] = []

    def loader(name: str):
        names.append(name)
        return zone

    assert resolve_user_timezone({"MIST_TIMEZONE": "Test/Zone"}, load_zone=loader) is zone
    assert names == ["Test/Zone"]
