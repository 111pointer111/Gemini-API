import unittest
from unittest.mock import patch

from gemini_webapi.utils.browser_session import load_full_browser_cookie_session


def cookie(name: str, value: str = "secret") -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "domain": ".google.com",
        "path": "/",
        "expires": None,
    }


class BrowserCookieSessionTest(unittest.TestCase):
    @patch("gemini_webapi.utils.browser_session.load_browser_cookies")
    def test_prefers_complete_chrome_session(self, load_cookies) -> None:
        load_cookies.return_value = {
            "safari": [cookie("__Secure-1PSID")],
            "chrome": [
                cookie("__Secure-1PSID"),
                cookie("__Secure-1PSIDTS"),
                cookie("SID"),
            ],
        }

        result = load_full_browser_cookie_session()

        assert result is not None
        assert result.browser == "chrome"
        assert result.cookie_count == 3
        assert result.cookies.get("SID") == "secret"

    @patch("gemini_webapi.utils.browser_session.load_browser_cookies")
    def test_honors_requested_browser(self, load_cookies) -> None:
        complete = [cookie("__Secure-1PSID"), cookie("__Secure-1PSIDTS")]
        load_cookies.return_value = {"chrome": complete, "safari": complete}

        result = load_full_browser_cookie_session("safari")

        assert result is not None
        assert result.browser == "safari"

    @patch("gemini_webapi.utils.browser_session.load_browser_cookies")
    def test_rejects_incomplete_sessions(self, load_cookies) -> None:
        load_cookies.return_value = {"chrome": [cookie("__Secure-1PSID")]}

        assert load_full_browser_cookie_session() is None


if __name__ == "__main__":
    unittest.main()
