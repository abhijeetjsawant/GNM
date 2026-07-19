"""Application-level HMAC sealing for local artifact trust roots.

The key lives outside every served job/character directory with owner-only
permissions. This detects offline edits made through the application account's
normal artifact paths. A hosted deployment should replace the local key with a
KMS/HSM-backed signer; an operating-system administrator who can read process
memory or the key file remains outside this local threat boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any


INTEGRITY_SCHEMA = "autoanim.hmac-sha256.v1"


def _canonical(document: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class IntegritySigner:
    """Create and verify deterministic HMAC seals without exposing the key."""

    def __init__(self, key_path: str | Path):
        self.key_path = Path(key_path).resolve()
        self.key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.key_path.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".integrity-key-", dir=self.key_path.parent
            )
            temporary_path = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                os.write(descriptor, secrets.token_bytes(32))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                # Hard-link publication is atomic: concurrent first starts can
                # never observe a partially written key or replace a winner.
                os.link(temporary_path, self.key_path)
            except FileExistsError:
                pass
            finally:
                temporary_path.unlink(missing_ok=True)
        self.key_path.chmod(0o600)
        self._key = self.key_path.read_bytes()
        if len(self._key) != 32:
            raise RuntimeError("AutoAnim integrity key must contain exactly 32 bytes")
        self.key_id = hashlib.sha256(self._key).hexdigest()[:16]

    def sign(self, document: dict[str, Any]) -> dict[str, Any]:
        signed = {key: value for key, value in document.items() if key != "integrity"}
        signature = hmac.new(self._key, _canonical(signed), hashlib.sha256).hexdigest()
        signed["integrity"] = {
            "schema": INTEGRITY_SCHEMA,
            "key_id": self.key_id,
            "signature": signature,
        }
        return signed

    def verify(self, document: dict[str, Any]) -> bool:
        integrity = document.get("integrity")
        if not isinstance(integrity, dict):
            return False
        if (
            integrity.get("schema") != INTEGRITY_SCHEMA
            or integrity.get("key_id") != self.key_id
            or not isinstance(integrity.get("signature"), str)
        ):
            return False
        expected = hmac.new(
            self._key, _canonical(document), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, integrity["signature"])


__all__ = ["INTEGRITY_SCHEMA", "IntegritySigner"]
