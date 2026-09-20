#!/usr/bin/env python3
"""Verify full browser cookies and proxy access with a harmless Gemini request."""

from __future__ import annotations

import argparse
import asyncio
import os

from gemini_webapi import GeminiClient
from gemini_webapi.utils.browser_session import load_full_browser_cookie_session


def configured_proxy(explicit: str | None) -> str | None:
    """Resolve the proxy without exposing its value in output."""
    return (
        explicit
        or os.getenv("GEMINI_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("http_proxy")
        or os.getenv("HTTP_PROXY")
    )


async def probe(browser: str | None, proxy: str, model: str) -> None:
    """Initialize Gemini from a full in-memory browser session and send one request."""
    selection = await asyncio.to_thread(
        load_full_browser_cookie_session,
        browser,
        verbose=True,
    )
    if selection is None:
        raise SystemExit("No complete Gemini browser session found.")

    client = GeminiClient(proxy=proxy)
    client.cookies = selection.cookies
    try:
        await client.init(timeout=60, auto_refresh=False, verbose=False)
        output = await client.generate_content(
            "Reply with exactly COOKIE_PROBE_OK",
            model=model,
            temporary=True,
        )
        models = [item for item in client.list_models() or [] if item.is_available]
        print(f"browser={selection.browser}")
        print(f"cookie_count={selection.cookie_count}")
        print(f"proxy_enabled={'yes' if proxy else 'no'}")
        print(f"account_status={client.account_status.name}")
        print(f"available_models={len(models)}")
        print(f"response_ok={'yes' if output.text.strip() == 'COOKIE_PROBE_OK' else 'no'}")
    finally:
        await client.close()


def main() -> None:
    """Parse options and run the live probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Browser name, such as chrome or safari")
    parser.add_argument("--proxy", help="Proxy URL; defaults to proxy environment variables")
    parser.add_argument("--model", default="gemini-pro", help="Gemini model name")
    args = parser.parse_args()

    proxy = configured_proxy(args.proxy)
    if not proxy:
        raise SystemExit("Proxy required. Set GEMINI_PROXY or HTTPS_PROXY first.")
    asyncio.run(probe(args.browser, proxy, args.model))


if __name__ == "__main__":
    main()
