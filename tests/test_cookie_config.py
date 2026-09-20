import stat
import tempfile
import unittest
from pathlib import Path

from gemini_webapi.utils.cookie_config import (
    cookies_from_mapping,
    cookies_to_mapping,
    load_cookie_config,
    parse_cookie_header,
    save_cookie_config,
)


class CookieConfigTest(unittest.TestCase):
    def test_parse_complete_cookie_header_preserves_equals(self) -> None:
        result = parse_cookie_header("Cookie: __Secure-1PSID=abc==; __Secure-1PSIDTS=def=; NID=123")
        assert result["__Secure-1PSID"] == "abc=="
        assert result["__Secure-1PSIDTS"] == "def="
        assert result["NID"] == "123"

    def test_parse_rejects_incomplete_cookie_header(self) -> None:
        caught: ValueError | None = None
        try:
            parse_cookie_header("__Secure-1PSID=abc")
        except ValueError as exc:
            caught = exc
        assert caught is not None
        assert "__Secure-1PSIDTS" in str(caught)

    def test_save_load_and_cookie_jar_roundtrip(self) -> None:
        values = {
            "__Secure-1PSID": "session-value",
            "__Secure-1PSIDTS": "timestamp-value",
            "NID": "extra-value",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".gemini-cookie-session.json"
            save_cookie_config(path, values)

            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert load_cookie_config(path) == values
            assert cookies_to_mapping(cookies_from_mapping(values)) == values


if __name__ == "__main__":
    unittest.main()
