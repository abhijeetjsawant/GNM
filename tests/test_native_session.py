from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import autoanim_gnm.cli as cli
from autoanim_gnm.api import create_app
from autoanim_gnm.cli import build_parser
from autoanim_gnm.errors import AutoAnimError


TOKEN = secrets.token_hex(32)


def _client(tmp_path: Path, *, token: str | None = TOKEN) -> TestClient:
    return TestClient(
        create_app(
            tmp_path / "jobs",
            model_path=tmp_path / "missing.task",
            session_token=token,
        )
    )


def test_native_session_rejects_missing_and_wrong_credentials(tmp_path: Path) -> None:
    client = _client(tmp_path)

    missing = client.get("/api/health")
    wrong = client.get("/api/health", headers={"X-AutoAnim-Token": secrets.token_hex(32)})

    for response in (missing, wrong):
        assert response.status_code == 401
        assert response.json() == {
            "code": "SESSION_UNAUTHORIZED",
            "message": "A valid native session credential is required",
            "details": {},
            "retryable": False,
        }
        assert response.headers["cache-control"] == "no-store"


def test_native_session_rejects_ambiguous_duplicate_header(tmp_path: Path) -> None:
    response = _client(tmp_path).get(
        "/api/health",
        headers=[
            ("X-AutoAnim-Token", TOKEN),
            ("X-AutoAnim-Token", TOKEN),
        ],
    )

    assert response.status_code == 401
    assert response.json()["code"] == "SESSION_UNAUTHORIZED"


def test_native_session_accepts_header_or_cookie_on_every_route(tmp_path: Path) -> None:
    client = _client(tmp_path)

    header = client.get("/api/health", headers={"X-AutoAnim-Token": TOKEN})
    client.cookies.set("autoanim_session", TOKEN)
    cookie = client.get("/")
    protected_schema = client.get("/openapi.json")

    assert header.status_code == 200
    assert cookie.status_code == 200
    assert protected_schema.status_code == 200


def test_native_session_rejects_non_loopback_and_wrong_bound_port(tmp_path: Path) -> None:
    client = _client(tmp_path)
    credential = {"X-AutoAnim-Token": TOKEN}

    remote = client.get("/api/health", headers={**credential, "Host": "example.com"})
    wrong_port = client.get(
        "/api/health", headers={**credential, "Host": "127.0.0.1:8765"}
    )
    oversized_port = client.get(
        "/api/health", headers={**credential, "Host": f"localhost:{'9' * 5000}"}
    )

    for response in (remote, wrong_port, oversized_port):
        assert response.status_code == 400
        assert response.json()["code"] == "HOST_INVALID"


@pytest.mark.parametrize(
    "base_url",
    (
        "http://localhost:8765",
        "http://127.0.0.1:8765",
        "http://testserver:8765",
    ),
)
def test_native_session_accepts_allowlisted_host_with_bound_port(
    tmp_path: Path, base_url: str
) -> None:
    client = TestClient(
        create_app(
            tmp_path / "jobs",
            model_path=tmp_path / "missing.task",
            session_token=TOKEN,
        ),
        base_url=base_url,
    )

    response = client.get("/api/health", headers={"X-AutoAnim-Token": TOKEN})

    assert response.status_code == 200


def test_native_session_accepts_ipv6_loopback_host(tmp_path: Path) -> None:
    # Starlette's in-process transport cannot parse an IPv6 base_url, so keep
    # its bound test port and exercise the equivalent bracketed Host directly.
    client = _client(tmp_path)

    response = client.get(
        "/api/health",
        headers={"Host": "[::1]:80", "X-AutoAnim-Token": TOKEN},
    )

    assert response.status_code == 200


def test_no_token_preserves_unauthenticated_development_behavior(tmp_path: Path) -> None:
    client = _client(tmp_path, token=None)

    assert client.get("/api/health", headers={"Host": "example.com"}).status_code == 200
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("weak_token", ("", "short", "a" * 32, "f" * 63))
def test_session_token_strength_fails_closed(tmp_path: Path, weak_token: str) -> None:
    with pytest.raises(AutoAnimError) as error:
        create_app(
            tmp_path / "jobs",
            model_path=tmp_path / "missing.task",
            session_token=weak_token,
        )
    assert error.value.code == "INPUT_INVALID"


def test_serve_parser_accepts_session_token() -> None:
    args = build_parser().parse_args(["serve", "--session-token", TOKEN])

    assert args.session_token == TOKEN


def test_serve_forwards_session_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_create_app(*args: object, **kwargs: object) -> object:
        observed["create_args"] = args
        observed["create_kwargs"] = kwargs
        return object()

    def fake_run(app: object, **kwargs: object) -> None:
        observed["app"] = app
        observed["run_kwargs"] = kwargs

    monkeypatch.setattr(cli, "ApplicationService", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    exit_code = cli.main(
        [
            "serve",
            "--artifacts",
            str(tmp_path / "jobs"),
            "--port",
            "8765",
            "--session-token",
            TOKEN,
        ]
    )

    assert exit_code == 0
    assert observed["create_kwargs"]["session_token"] == TOKEN
    assert observed["run_kwargs"] == {
        "host": "127.0.0.1",
        "port": 8765,
        "workers": 1,
    }
