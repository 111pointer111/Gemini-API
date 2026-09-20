"""Parse and persist a manually copied Google Cookie header."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from curl_cffi.requests import Cookies

REQUIRED_GEMINI_COOKIES = frozenset({"__Secure-1PSID", "__Secure-1PSIDTS"})
_COOKIE_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def parse_cookie_header(raw_header: str) -> dict[str, str]:
    """Parse a complete request Cookie header copied from browser DevTools.

    Args:
        raw_header: Cookie header value, optionally prefixed with ``Cookie:``.

    Returns:
        A mapping containing every cookie name and value.

    Raises:
        ValueError: If the header is malformed or lacks required Gemini cookies.
    """
    normalized = raw_header.strip()
    if normalized[:7].lower() == "cookie:":
        normalized = normalized[7:].strip()
    if not normalized:
        raise ValueError("Cookie 内容为空。")
    if "\r" in normalized or "\n" in normalized:
        raise ValueError("Cookie 不能包含换行, 请只复制 Request Headers 中的 Cookie 值。")

    cookies: dict[str, str] = {}
    for segment in normalized.split(";"):
        name, separator, value = segment.strip().partition("=")
        if not separator or not name or not _COOKIE_NAME.fullmatch(name):
            raise ValueError(f"Cookie 片段格式不正确: {name or '[空]'}")
        cookies[name] = value.strip()

    missing = REQUIRED_GEMINI_COOKIES - cookies.keys()
    if missing:
        names = "、".join(sorted(missing))
        raise ValueError(f"缺少 Gemini 登录 Cookie: {names}")
    return cookies


def cookies_from_mapping(values: dict[str, str]) -> Cookies:
    """Build a secure Google cookie jar from a name-value mapping."""
    jar = Cookies()
    for name, value in values.items():
        jar.set(name, value, domain=".google.com", path="/", secure=True)
    return jar


def cookies_to_mapping(jar: Cookies) -> dict[str, str]:
    """Return only Google cookie values from a live cookie jar."""
    return {
        cookie.name: cookie.value
        for cookie in jar.jar
        if cookie.value is not None
        and (cookie.domain == "google.com" or cookie.domain.endswith(".google.com"))
    }


def load_cookie_config(path: Path) -> dict[str, str] | None:
    """Load a valid ignored local cookie configuration without logging values."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("cookies")
        if not isinstance(values, dict):
            return None
        cookies = {str(name): str(value) for name, value in values.items()}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return cookies if REQUIRED_GEMINI_COOKIES.issubset(cookies) else None


def save_cookie_config(path: Path, values: dict[str, str]) -> None:
    """Atomically save cookies in a user-only local file.

    Raises:
        ValueError: If required Gemini cookies are missing.
        OSError: If the file cannot be written or secured.
    """
    missing = REQUIRED_GEMINI_COOKIES - values.keys()
    if missing:
        raise ValueError("Cannot persist an incomplete Gemini session.")
    payload = json.dumps(
        {"saved_at": datetime.now(UTC).isoformat(), "cookies": values},
        ensure_ascii=False,
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)
