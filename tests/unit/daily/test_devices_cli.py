"""Tests for `scripts/daily/devices.py`, the room-device provisioning CLI.

Every `openssl` call goes through the injected runner. The fake writes a placeholder file for each
`-out` and answers the certificate's SAN query, so no key or certificate is made, except in the
last tests, which run the real `openssl` in a temporary directory when one is on `PATH`.
Every directory lives under `tmp_path`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import socket
import ssl
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.daily import devices

NOW = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)
SERVER_IP = "192.168.1.20"
OPENSSL = "/usr/bin/openssl"
GIT_OPENSSL = "C:/Program Files/Git/usr/bin/openssl.exe"
NAME = "example-room"
AREA = "example-area"


def san_output(ip: str) -> bytes:
    return f"X509v3 Subject Alternative Name: \n    IP Address:{ip}\n".encode()


class FakeRunner:
    """Records each argv, writes a placeholder for its `-out`, and keeps each `-extfile`'s text.

    `fail_when` names a word: the first call holding it exits 1 and writes nothing.
    """

    def __init__(self, *, san_ip: str = SERVER_IP, fail_when: str | None = None):
        self.calls: list[list[str]] = []
        self.extfiles: list[str] = []
        self.san_ip = san_ip
        self.fail_when = fail_when

    def __call__(self, argv):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if self.fail_when is not None and self.fail_when in argv:
            self.fail_when = None
            return devices.CommandResult(1, b"", b"openssl: simulated failure")
        if "-extfile" in argv:
            self.extfiles.append(Path(argv[argv.index("-extfile") + 1]).read_text("utf-8"))
        if "-out" in argv:
            Path(argv[argv.index("-out") + 1]).write_bytes(f"placeholder {argv[1]}\n".encode())
        if "-ext" in argv and "subjectAltName" in argv:
            return devices.CommandResult(0, san_output(self.san_ip), b"")
        return devices.CommandResult(0, b"", b"")


def make_ctx(runner=None, *, environ=None, which=None, git_openssl=None) -> devices.Context:
    return devices.Context(
        runner=runner if runner is not None else FakeRunner(),
        environ=environ if environ is not None else {},
        clock=lambda: NOW,
        which=which if which is not None else (lambda name: OPENSSL),
        git_openssl=Path(git_openssl) if git_openssl is not None else Path(GIT_OPENSSL),
    )


def run(ctx: devices.Context, *argv: str) -> int:
    return devices.main(list(argv), ctx)


@pytest.fixture
def device_dir(tmp_path) -> Path:
    return (tmp_path / "root" / "devices").resolve()


@pytest.fixture
def ca_dir(tmp_path) -> Path:
    return (tmp_path / "root" / "devices-ca").resolve()


def init(ctx, device_dir: Path, ca_dir: Path, ip: str = SERVER_IP) -> int:
    return run(ctx, "init", "--dir", str(device_dir), "--ca-dir", str(ca_dir), "--ip", ip)


@pytest.fixture
def initialised(device_dir, ca_dir) -> FakeRunner:
    runner = FakeRunner()
    assert init(make_ctx(runner), device_dir, ca_dir) == 0
    return runner


def add(ctx, device_dir: Path, out: Path, name: str = NAME, area: str = AREA) -> int:
    # The name goes after `--`, so one starting with a hyphen reaches the name check.
    return run(ctx, "add", "--area", area, "--out", str(out), "--dir", str(device_dir), "--", name)


def read_registry(device_dir: Path) -> dict:
    return json.loads((device_dir / "registry.json").read_text(encoding="utf-8"))


def read_secrets(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and ":" in line:
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip().strip('"')
    return values


def secret_of(token: str) -> bytes:
    encoded = token.partition(".")[2]
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))


def files_under(path: Path) -> set[str]:
    return {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}


# --- init ------------------------------------------------------------------------------------


def test_init_makes_an_ec_p256_ca_and_a_server_certificate_with_the_ip_san(
    device_dir, ca_dir, initialised
):
    tls = device_dir / "tls"
    assert initialised.calls == [
        [
            OPENSSL,
            "genpkey",
            "-algorithm",
            "EC",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-out",
            str(ca_dir / "ca.key"),
        ],
        [
            OPENSSL,
            "req",
            "-x509",
            "-new",
            "-key",
            str(ca_dir / "ca.key"),
            "-sha256",
            "-days",
            "3650",
            "-subj",
            "/CN=MIST device CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-out",
            str(tls / "ca.pem"),
        ],
        [
            OPENSSL,
            "genpkey",
            "-algorithm",
            "EC",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-out",
            str(tls / "server.key.new"),
        ],
        [
            OPENSSL,
            "req",
            "-new",
            "-key",
            str(tls / "server.key.new"),
            "-subj",
            f"/CN={SERVER_IP}",
            "-out",
            str(tls / "server.csr"),
        ],
        [
            OPENSSL,
            "x509",
            "-req",
            "-in",
            str(tls / "server.csr"),
            "-CA",
            str(tls / "ca.pem"),
            "-CAkey",
            str(ca_dir / "ca.key"),
            "-set_serial",
            f"0x{int(NOW.timestamp() * 1_000_000):x}",
            "-days",
            "3650",
            "-sha256",
            "-extfile",
            str(tls / "server.ext"),
            "-out",
            str(tls / "server.pem.new"),
        ],
    ]
    assert initialised.extfiles == [
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid:always\n"
        f"subjectAltName=IP:{SERVER_IP}\n"
    ]


def test_init_writes_the_ca_key_only_to_the_ca_dir_and_the_server_key_only_to_the_devices_dir(
    device_dir, ca_dir, initialised
):
    assert files_under(device_dir) == {
        "registry.json",
        "tls/ca.pem",
        "tls/server.pem",
        "tls/server.key",
    }
    assert files_under(ca_dir) == {"ca.key"}


def test_init_writes_an_empty_registry(device_dir, initialised):
    assert read_registry(device_dir) == {"version": 1, "devices": []}


def test_init_defaults_both_dirs_from_the_instance_root(tmp_path):
    root = (tmp_path / "instance").resolve()
    runner = FakeRunner()
    ctx = make_ctx(runner, environ={"MIST_DAILY_ROOT": str(root)})

    assert run(ctx, "init", "--ip", SERVER_IP) == 0

    assert (root / "devices" / "registry.json").is_file()
    assert files_under(root / "devices-ca") == {"ca.key"}


def test_mist_device_dir_wins_over_the_instance_root(tmp_path):
    root = (tmp_path / "instance").resolve()
    elsewhere = (tmp_path / "elsewhere" / "devices").resolve()
    ctx = make_ctx(environ={"MIST_DAILY_ROOT": str(root), "MIST_DEVICE_DIR": str(elsewhere)})

    assert run(ctx, "init", "--ip", SERVER_IP) == 0

    assert (elsewhere / "registry.json").is_file()
    assert not (root / "devices").exists()


def test_init_refuses_without_a_devices_dir_or_instance_root(tmp_path, capsys):
    runner = FakeRunner()
    assert run(make_ctx(runner), "init", "--ip", SERVER_IP) == 2
    assert runner.calls == []
    assert "--dir" in capsys.readouterr().err


def assert_refused_and_nothing_written(code: int, runner: FakeRunner, *paths: Path) -> None:
    assert code == 2
    assert runner.calls == []
    for path in paths:
        assert not path.exists(), path


@pytest.mark.parametrize(
    "where",
    [
        pytest.param("inside", id="ca-dir-inside-devices-dir"),
        pytest.param("same", id="ca-dir-is-devices-dir"),
    ],
)
def test_init_refuses_a_ca_dir_inside_the_devices_dir(device_dir, where, capsys):
    ca = device_dir / "ca" if where == "inside" else device_dir
    runner = FakeRunner()

    code = init(make_ctx(runner), device_dir, ca)

    assert_refused_and_nothing_written(code, runner, device_dir)
    assert "inside the devices dir" in capsys.readouterr().err


def test_init_refuses_a_ca_dir_inside_a_git_work_tree(tmp_path, device_dir, capsys):
    repo = (tmp_path / "repo").resolve()
    (repo / ".git").mkdir(parents=True)
    runner = FakeRunner()

    code = init(make_ctx(runner), device_dir, repo / "nested" / "devices-ca")

    assert_refused_and_nothing_written(code, runner, device_dir, repo / "nested")
    assert "git work tree" in capsys.readouterr().err


def test_init_refuses_a_devices_dir_inside_a_git_work_tree(tmp_path, ca_dir):
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    # A linked worktree holds `.git` as a file.
    (repo / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    runner = FakeRunner()

    code = init(make_ctx(runner), repo / "devices", ca_dir)

    assert_refused_and_nothing_written(code, runner, repo / "devices", ca_dir)


@pytest.mark.parametrize("ip", ["0.0.0.0", "8.8.8.8", "127.0.0.1", "192.168.1", "not-an-ip"])
def test_init_refuses_an_ip_that_is_not_a_private_ipv4_address(device_dir, ca_dir, ip):
    runner = FakeRunner()
    assert_refused_and_nothing_written(
        init(make_ctx(runner), device_dir, ca_dir, ip), runner, device_dir, ca_dir
    )


def test_init_refuses_a_second_run_and_leaves_the_ca_alone(device_dir, ca_dir, initialised):
    before = (ca_dir / "ca.key").read_bytes()
    runner = FakeRunner()

    assert init(make_ctx(runner), device_dir, ca_dir) == 2

    assert runner.calls == []
    assert (ca_dir / "ca.key").read_bytes() == before


def test_init_uses_the_git_for_windows_openssl_when_none_is_on_path(tmp_path, device_dir, ca_dir):
    git_openssl = tmp_path / "Git" / "usr" / "bin" / "openssl.exe"
    git_openssl.parent.mkdir(parents=True)
    git_openssl.write_bytes(b"")
    runner = FakeRunner()
    ctx = make_ctx(runner, which=lambda name: None, git_openssl=git_openssl)

    assert init(ctx, device_dir, ca_dir) == 0

    assert {call[0] for call in runner.calls} == {str(git_openssl)}


def test_init_refuses_naming_both_places_when_no_openssl_is_found(device_dir, ca_dir, capsys):
    runner = FakeRunner()
    ctx = make_ctx(runner, which=lambda name: None)

    code = init(ctx, device_dir, ca_dir)

    assert_refused_and_nothing_written(code, runner, device_dir, ca_dir)
    err = capsys.readouterr().err
    assert "PATH" in err and GIT_OPENSSL in err


def test_init_names_the_failed_step_and_exits_1(device_dir, ca_dir, capsys):
    runner = FakeRunner(fail_when="x509")

    assert init(make_ctx(runner), device_dir, ca_dir) == 1

    err = capsys.readouterr().err
    assert "simulated failure" in err
    assert not (device_dir / "registry.json").exists()


# --- add -------------------------------------------------------------------------------------


@pytest.fixture
def out(tmp_path) -> Path:
    return (tmp_path / "private" / NAME).resolve()


def test_add_writes_the_secrets_and_ca_into_the_private_directory(device_dir, initialised, out):
    runner = FakeRunner()

    assert add(make_ctx(runner), device_dir, out) == 0

    assert files_under(out) == {"secrets.yaml", "mist_ca.pem"}
    assert (out / "mist_ca.pem").read_bytes() == (device_dir / "tls" / "ca.pem").read_bytes()
    values = read_secrets(out / "secrets.yaml")
    assert values["mist_device_id"] == NAME
    assert values["mist_device_token"].startswith(f"{NAME}.")
    assert len(secret_of(values["mist_device_token"])) == 32
    assert values["mist_server_host"] == SERVER_IP
    assert values["mist_server_port"] == "8443"
    assert len(values["mist_ota_password"]) == 32
    # The host comes from the server certificate's SAN.
    assert runner.calls == [
        [
            OPENSSL,
            "x509",
            "-in",
            str(device_dir / "tls" / "server.pem"),
            "-noout",
            "-ext",
            "subjectAltName",
        ]
    ]


def test_add_keeps_only_the_secrets_sha256_in_the_registry(device_dir, initialised, out):
    assert add(make_ctx(), device_dir, out) == 0

    token = read_secrets(out / "secrets.yaml")["mist_device_token"]
    registry_text = (device_dir / "registry.json").read_text(encoding="utf-8")
    assert read_registry(device_dir)["devices"] == [
        {
            "name": NAME,
            "area": AREA,
            "token_sha256": hashlib.sha256(secret_of(token)).hexdigest(),
            "revoked": False,
            "created": "2026-11-17T09:00:00Z",
        }
    ]
    assert token not in registry_text
    assert token.partition(".")[2] not in registry_text


def test_add_never_prints_or_passes_on_the_token(device_dir, initialised, out, capsys):
    runner = FakeRunner()
    capsys.readouterr()

    assert add(make_ctx(runner), device_dir, out) == 0

    token = read_secrets(out / "secrets.yaml")["mist_device_token"]
    secret = token.partition(".")[2]
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert secret not in stream
    assert NAME in captured.out
    assert all(secret not in " ".join(call) for call in runner.calls)
    assert files_under(device_dir) == {
        "registry.json",
        "tls/ca.pem",
        "tls/server.pem",
        "tls/server.key",
    }


def test_add_refuses_a_private_directory_inside_a_git_work_tree(
    tmp_path, device_dir, initialised, capsys
):
    repo = (tmp_path / "repo").resolve()
    (repo / ".git").mkdir(parents=True)
    before = (device_dir / "registry.json").read_bytes()
    runner = FakeRunner()

    code = add(make_ctx(runner), device_dir, repo / "private" / NAME)

    assert_refused_and_nothing_written(code, runner, repo / "private")
    assert (device_dir / "registry.json").read_bytes() == before
    assert "git work tree" in capsys.readouterr().err


@pytest.mark.parametrize("inside", ["devices", "ca"])
def test_add_refuses_a_private_directory_inside_the_devices_or_ca_dir(
    device_dir, ca_dir, initialised, inside
):
    parent = device_dir if inside == "devices" else ca_dir
    runner = FakeRunner()
    code = run(
        make_ctx(runner),
        "add",
        NAME,
        "--area",
        AREA,
        "--out",
        str(parent / "private"),
        "--dir",
        str(device_dir),
        "--ca-dir",
        str(ca_dir),
    )
    assert_refused_and_nothing_written(code, runner, parent / "private")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("Example", id="uppercase"),
        pytest.param("1room", id="leading-digit"),
        pytest.param("-room", id="leading-hyphen"),
        pytest.param("room_1", id="underscore"),
        pytest.param("a" * 25, id="25-characters"),
        pytest.param("desktop", id="reserved-desktop"),
        pytest.param("all", id="reserved-all"),
        pytest.param("room", id="reserved-room"),
        pytest.param("", id="empty"),
    ],
)
def test_add_refuses_a_name_off_the_rules(device_dir, initialised, out, name):
    runner = FakeRunner()
    assert_refused_and_nothing_written(
        add(make_ctx(runner), device_dir, out, name=name), runner, out
    )
    assert read_registry(device_dir)["devices"] == []


@pytest.mark.parametrize("name", ["a", "a" * 24, "b2-x"])
def test_add_accepts_a_name_on_the_rules(device_dir, initialised, out, name):
    assert add(make_ctx(), device_dir, out, name=name) == 0


@pytest.mark.parametrize("area", ["", "  ", "bad\narea", "x" * 65])
def test_add_refuses_an_unusable_area(device_dir, initialised, out, area):
    runner = FakeRunner()
    assert_refused_and_nothing_written(
        add(make_ctx(runner), device_dir, out, area=area), runner, out
    )


def test_add_refuses_a_name_already_registered(device_dir, initialised, out, tmp_path):
    assert add(make_ctx(), device_dir, out) == 0
    runner = FakeRunner()

    code = add(make_ctx(runner), device_dir, tmp_path / "private" / "second")

    assert_refused_and_nothing_written(code, runner, tmp_path / "private" / "second")
    assert len(read_registry(device_dir)["devices"]) == 1


def test_add_refuses_before_init(device_dir, out, capsys):
    runner = FakeRunner()
    assert_refused_and_nothing_written(add(make_ctx(runner), device_dir, out), runner, out)
    assert "init" in capsys.readouterr().err


def test_add_keeps_lines_raj_added_to_an_existing_secrets_file(device_dir, initialised, out):
    out.mkdir(parents=True)
    (out / "secrets.yaml").write_text(
        'wifi_ssid: "example-network"\nwifi_password: "example-password"\n', encoding="utf-8"
    )

    assert add(make_ctx(), device_dir, out) == 0

    values = read_secrets(out / "secrets.yaml")
    assert values["wifi_ssid"] == "example-network"
    assert values["wifi_password"] == "example-password"
    assert values["mist_device_id"] == NAME


# --- list, revoke, rotate --------------------------------------------------------------------


def test_list_prints_each_device_without_its_hash(device_dir, initialised, out, capsys):
    assert add(make_ctx(), device_dir, out) == 0
    digest = read_registry(device_dir)["devices"][0]["token_sha256"]
    capsys.readouterr()

    assert run(make_ctx(), "list", "--dir", str(device_dir)) == 0

    printed = capsys.readouterr().out
    assert printed.splitlines() == [f"{NAME}  area {AREA}  active  created 2026-11-17T09:00:00Z"]
    assert digest not in printed


def test_revoke_marks_the_device_revoked(device_dir, initialised, out, capsys):
    assert add(make_ctx(), device_dir, out) == 0
    runner = FakeRunner()

    assert run(make_ctx(runner), "revoke", NAME, "--dir", str(device_dir)) == 0

    assert read_registry(device_dir)["devices"][0]["revoked"] is True
    assert runner.calls == []
    capsys.readouterr()
    run(make_ctx(), "list", "--dir", str(device_dir))
    assert "revoked" in capsys.readouterr().out


def test_revoke_refuses_an_unknown_name(device_dir, initialised):
    before = (device_dir / "registry.json").read_bytes()
    assert run(make_ctx(), "revoke", "example-none", "--dir", str(device_dir)) == 2
    assert (device_dir / "registry.json").read_bytes() == before


def test_rotate_replaces_the_token_and_keeps_the_rest_of_the_secrets_file(
    device_dir, initialised, out, capsys
):
    assert add(make_ctx(), device_dir, out) == 0
    with (out / "secrets.yaml").open("a", encoding="utf-8") as handle:
        handle.write('wifi_ssid: "example-network"\n')
    before = read_secrets(out / "secrets.yaml")
    old_digest = read_registry(device_dir)["devices"][0]["token_sha256"]
    capsys.readouterr()

    assert run(make_ctx(), "rotate", NAME, "--out", str(out), "--dir", str(device_dir)) == 0

    after = read_secrets(out / "secrets.yaml")
    assert after["mist_device_token"] != before["mist_device_token"]
    assert {k: v for k, v in after.items() if k != "mist_device_token"} == {
        k: v for k, v in before.items() if k != "mist_device_token"
    }
    record = read_registry(device_dir)["devices"][0]
    assert (
        record["token_sha256"] == hashlib.sha256(secret_of(after["mist_device_token"])).hexdigest()
    )
    assert record["token_sha256"] != old_digest
    captured = capsys.readouterr()
    secret = after["mist_device_token"].partition(".")[2]
    assert secret not in captured.out and secret not in captured.err


def test_rotate_says_a_session_on_the_old_token_is_closed_and_where_to_reflash_from(
    device_dir, initialised, out, capsys
):
    assert add(make_ctx(), device_dir, out) == 0
    capsys.readouterr()

    assert run(make_ctx(), "rotate", NAME, "--out", str(out), "--dir", str(device_dir)) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"rotated {NAME}'s token; a session still on the old one is closed within 10 s; "
        f"reflash the device from {out / 'secrets.yaml'}"
    ]


def test_rotate_refuses_a_revoked_device(device_dir, initialised, out):
    assert add(make_ctx(), device_dir, out) == 0
    assert run(make_ctx(), "revoke", NAME, "--dir", str(device_dir)) == 0
    before = (out / "secrets.yaml").read_bytes()

    assert run(make_ctx(), "rotate", NAME, "--out", str(out), "--dir", str(device_dir)) == 2

    assert (out / "secrets.yaml").read_bytes() == before


def test_rotate_refuses_an_unknown_name(device_dir, initialised, out):
    assert (
        run(make_ctx(), "rotate", "example-none", "--out", str(out), "--dir", str(device_dir)) == 2
    )
    assert not out.exists()


def test_rotate_refuses_a_private_directory_inside_a_git_work_tree(
    tmp_path, device_dir, initialised, out
):
    assert add(make_ctx(), device_dir, out) == 0
    before = (device_dir / "registry.json").read_bytes()
    repo = (tmp_path / "repo").resolve()
    (repo / ".git").mkdir(parents=True)

    code = run(make_ctx(), "rotate", NAME, "--out", str(repo / NAME), "--dir", str(device_dir))

    assert code == 2
    assert not (repo / NAME).exists()
    assert (device_dir / "registry.json").read_bytes() == before


# --- cert ------------------------------------------------------------------------------------


def test_cert_reissues_the_server_certificate_for_a_new_ip(device_dir, ca_dir, initialised):
    runner = FakeRunner()
    new_ip = "10.0.0.7"

    code = run(
        make_ctx(runner), "cert", "--ip", new_ip, "--ca-dir", str(ca_dir), "--dir", str(device_dir)
    )

    assert code == 0
    assert [call[1] for call in runner.calls] == ["genpkey", "req", "x509"]
    assert runner.calls[2][runner.calls[2].index("-CAkey") + 1] == str(ca_dir / "ca.key")
    assert runner.extfiles[0].endswith(f"subjectAltName=IP:{new_ip}\n")
    assert files_under(device_dir) == {
        "registry.json",
        "tls/ca.pem",
        "tls/server.pem",
        "tls/server.key",
    }
    assert files_under(ca_dir) == {"ca.key"}


def test_cert_refuses_a_ca_dir_inside_the_devices_dir(device_dir, initialised):
    runner = FakeRunner()
    code = run(
        make_ctx(runner),
        "cert",
        "--ip",
        SERVER_IP,
        "--ca-dir",
        str(device_dir / "ca"),
        "--dir",
        str(device_dir),
    )
    assert code == 2 and runner.calls == []


def test_cert_refuses_without_the_ca_key(device_dir, ca_dir, initialised):
    (ca_dir / "ca.key").unlink()
    runner = FakeRunner()
    code = run(
        make_ctx(runner),
        "cert",
        "--ip",
        SERVER_IP,
        "--ca-dir",
        str(ca_dir),
        "--dir",
        str(device_dir),
    )
    assert code == 2 and runner.calls == []


def test_cert_keeps_the_old_certificate_when_signing_fails(device_dir, ca_dir, initialised):
    tls = device_dir / "tls"
    before = {name: (tls / name).read_bytes() for name in ("server.pem", "server.key")}
    runner = FakeRunner(fail_when="x509")

    code = run(
        make_ctx(runner),
        "cert",
        "--ip",
        "10.0.0.7",
        "--ca-dir",
        str(ca_dir),
        "--dir",
        str(device_dir),
    )

    assert code == 1
    assert {name: (tls / name).read_bytes() for name in before} == before
    assert files_under(device_dir) == {
        "registry.json",
        "tls/ca.pem",
        "tls/server.pem",
        "tls/server.key",
    }


# --- With the real openssl -------------------------------------------------------------------

real_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="no openssl on PATH")


def real_ctx() -> devices.Context:
    return devices.Context(environ={}, clock=lambda: datetime.now(UTC))


def handshake(ca_pem: Path, server_pem: Path, server_key: Path, server_name: str) -> str:
    """Complete one TLS handshake over a socket pair; returns the client's view of the cipher."""
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(server_pem, server_key)
    client_ctx = ssl.create_default_context(cafile=str(ca_pem))
    left, right = socket.socketpair()
    errors: list[BaseException] = []

    def serve():
        try:
            with server_ctx.wrap_socket(right, server_side=True) as tls:
                tls.recv(1)
        except (OSError, ssl.SSLError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with client_ctx.wrap_socket(left, server_hostname=server_name) as tls:
            cipher = tls.cipher()[0]
            tls.sendall(b"x")
    finally:
        thread.join(5)
    assert not errors, errors
    return cipher


@real_openssl
def test_the_real_openssl_makes_a_chain_a_client_accepts_for_the_ip(tmp_path):
    device_dir = (tmp_path / "devices").resolve()
    ca_dir = (tmp_path / "devices-ca").resolve()

    assert init(real_ctx(), device_dir, ca_dir) == 0

    tls = device_dir / "tls"
    assert files_under(device_dir) == {
        "registry.json",
        "tls/ca.pem",
        "tls/server.pem",
        "tls/server.key",
    }
    assert files_under(ca_dir) == {"ca.key"}
    assert handshake(tls / "ca.pem", tls / "server.pem", tls / "server.key", SERVER_IP)
    with pytest.raises(ssl.SSLCertVerificationError):
        handshake(tls / "ca.pem", tls / "server.pem", tls / "server.key", "192.168.1.21")


@real_openssl
def test_the_real_openssl_san_read_gives_add_the_server_host(tmp_path):
    device_dir = (tmp_path / "devices").resolve()
    ca_dir = (tmp_path / "devices-ca").resolve()
    out = (tmp_path / "private" / NAME).resolve()
    assert init(real_ctx(), device_dir, ca_dir) == 0

    assert add(real_ctx(), device_dir, out) == 0

    assert read_secrets(out / "secrets.yaml")["mist_server_host"] == SERVER_IP
