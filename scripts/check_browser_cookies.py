#!/usr/bin/env python3
"""Safely report browser cookie availability without printing cookie values."""

from __future__ import annotations

import argparse

from gemini_webapi.utils.load_browser_cookies import load_browser_cookies

REQUIRED = {"__Secure-1PSID", "__Secure-1PSIDTS"}


def main() -> None:
    """List browsers that can supply a Gemini Web session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-names",
        action="store_true",
        help="Show cookie names. Values are never printed.",
    )
    args = parser.parse_args()

    browser_cookies = load_browser_cookies(domain_name="google.com", verbose=True)
    if not browser_cookies:
        raise SystemExit("No readable Google cookies found. Log in to Gemini in a browser first.")

    usable = False
    for browser, cookies in sorted(browser_cookies.items()):
        names = sorted({cookie["name"] for cookie in cookies})
        ready = REQUIRED.issubset(names)
        usable = usable or ready
        print(f"{browser}: count={len(names)}, gemini_ready={'yes' if ready else 'no'}")
        if args.show_names:
            print("  names=" + ", ".join(names))

    if not usable:
        raise SystemExit("Cookies were found, but no browser has a complete Gemini session.")


if __name__ == "__main__":
    main()
