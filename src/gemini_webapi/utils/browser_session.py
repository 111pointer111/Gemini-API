"""Load a complete Google browser session without persisting cookie values."""

from __future__ import annotations

from dataclasses import dataclass

from curl_cffi.requests import Cookies

from .load_browser_cookies import load_browser_cookies

_REQUIRED_COOKIES = {"__Secure-1PSID", "__Secure-1PSIDTS"}
_BROWSER_PRIORITY = (
    "chrome",
    "safari",
    "firefox",
    "edge",
    "brave",
    "chromium",
    "vivaldi",
    "opera",
    "opera_gx",
    "librewolf",
)


@dataclass(frozen=True, slots=True)
class BrowserCookieSession:
    """A selected browser and its in-memory Google cookie jar."""

    browser: str
    cookies: Cookies
    cookie_names: tuple[str, ...]

    @property
    def cookie_count(self) -> int:
        """Return the number of distinct cookie names loaded."""
        return len(self.cookie_names)


def load_full_browser_cookie_session(
    browser: str | None = None,
    *,
    verbose: bool = False,
) -> BrowserCookieSession | None:
    """Load all non-expired Google cookies from one authenticated browser.

    Cookie values stay in memory. The selected browser must contain both cookies
    required for an authenticated Gemini Web session.
    """
    browser_cookies = load_browser_cookies(domain_name="google.com", verbose=verbose)
    if not browser_cookies:
        return None

    requested = browser.strip().lower() if browser else None
    if requested:
        candidates = (requested,)
    else:
        known = [name for name in _BROWSER_PRIORITY if name in browser_cookies]
        candidates = (*known, *sorted(set(browser_cookies) - set(known)))

    for browser_name in candidates:
        cookie_list = browser_cookies.get(browser_name)
        if not cookie_list:
            continue

        cookie_names = {cookie["name"] for cookie in cookie_list}
        if not _REQUIRED_COOKIES.issubset(cookie_names):
            continue

        jar = Cookies()
        for cookie in cookie_list:
            jar.set(
                cookie["name"],
                cookie["value"],
                domain=cookie["domain"],
                path=cookie["path"],
                secure=True,
            )

        return BrowserCookieSession(
            browser=browser_name,
            cookies=jar,
            cookie_names=tuple(sorted(cookie_names)),
        )

    return None
