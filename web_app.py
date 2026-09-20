"""Small local web interface for chatting with Gemini and generating images."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import uvicorn
from curl_cffi import CurlFollow, CurlHttpVersion
from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gemini_webapi import ChatSession, GeminiClient, GeneratedImage, Image
from gemini_webapi.utils.browser_session import load_full_browser_cookie_session

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
GENERATED_DIR = WEB_DIR / "generated"
DEFAULT_MODEL = "gemini-pro"


class ChatRequest(BaseModel):
    """Incoming chat request from the browser."""

    message: Annotated[str, Field(min_length=1, max_length=12_000)]
    session_id: str | None = None
    model: str = DEFAULT_MODEL


class ImageRequest(BaseModel):
    """Incoming image-generation request from the browser."""

    prompt: Annotated[str, Field(min_length=1, max_length=4_000)]
    model: str = DEFAULT_MODEL


@dataclass(slots=True)
class SessionState:
    """A Gemini conversation and the lock protecting its turn order."""

    chat: ChatSession
    model: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WebRuntime:
    """Mutable runtime state shared by the FastAPI routes."""

    def __init__(self, client: GeminiClient) -> None:
        self.client = client
        self.sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str | None, model: str) -> tuple[str, SessionState]:
        """Return an existing conversation or create a new one."""
        if session_id and (state := self.sessions.get(session_id)) and state.model == model:
            return session_id, state

        new_id = uuid.uuid4().hex
        state = SessionState(chat=self.client.start_chat(model=model), model=model)
        self.sessions[new_id] = state
        return new_id, state


def configured_proxy() -> str | None:
    """Return the explicitly configured outbound proxy, if present."""
    return (
        os.getenv("GEMINI_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("http_proxy")
        or os.getenv("HTTP_PROXY")
    )


def is_google_image_url(url: str) -> bool:
    """Return whether a media URL is safe to expose or download."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == "googleusercontent.com" or hostname.endswith(".googleusercontent.com")
    )


async def save_generated_image(
    image: Image,
    client: GeminiClient,
    destination: Path,
    filename: str,
) -> Path:
    """Save one image, allowing the configured localhost proxy for trusted Google media."""
    if not is_google_image_url(image.url):
        raise ValueError("Gemini returned an unsupported image host.")

    if not isinstance(image, GeneratedImage):
        return Path(await image.save(path=str(destination), filename=filename))

    async with AsyncSession(
        impersonate=client.impersonate,
        cookies=client.cookies,
        proxy=configured_proxy(),
        allow_redirects=CurlFollow.ALL,
        http_version=CurlHttpVersion.NONE,
    ) as media_client:
        return Path(
            await image.save(
                path=str(destination),
                filename=filename,
                client=media_client,
                full_size=False,
            )
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize one long-lived Gemini client for the local web app."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    proxy = configured_proxy()
    if not proxy:
        raise RuntimeError("Proxy required. Set GEMINI_PROXY or HTTPS_PROXY before startup.")

    browser = os.getenv("GEMINI_BROWSER")
    browser_session = await asyncio.to_thread(
        load_full_browser_cookie_session,
        browser,
        verbose=False,
    )
    client = GeminiClient(proxy=proxy)
    if browser_session is not None:
        client.cookies = browser_session.cookies
    await client.init(timeout=60, auto_refresh=True, verbose=False)
    app.state.runtime = WebRuntime(client)
    app.state.cookie_browser = browser_session.browser if browser_session else None
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Gemini Canvas", version="1.0.0", lifespan=lifespan)


def runtime_from(request: Request) -> WebRuntime:
    """Read initialized runtime state or return a useful service error."""
    runtime: WebRuntime | None = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Gemini client is still starting.")
    return runtime


@app.get("/api/status")
async def status(request: Request) -> dict[str, Any]:
    """Return current account and model availability without exposing credentials."""
    runtime = runtime_from(request)
    models = [
        {"name": model.model_name, "label": model.display_name}
        for model in runtime.client.list_models() or []
        if model.is_available
    ]
    return {
        "ready": True,
        "account_status": runtime.client.account_status.name,
        "proxy_enabled": configured_proxy() is not None,
        "cookie_browser": getattr(request.app.state, "cookie_browser", None),
        "models": models,
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    """Send one turn while preserving per-browser conversation context."""
    runtime = runtime_from(request)
    session_id, state = runtime.get_session(payload.session_id, payload.model)
    try:
        async with state.lock:
            output = await state.chat.send_message(payload.message, temporary=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    return {
        "session_id": session_id,
        "text": output.text,
        "images": [image.url for image in output.images],
    }


@app.post("/api/images")
async def generate_image(payload: ImageRequest, request: Request) -> dict[str, Any]:
    """Generate images and save authenticated results for local browser display."""
    runtime = runtime_from(request)
    prompt = f"Generate an image from this description. Do not search the web: {payload.prompt}"
    try:
        output = await runtime.client.generate_content(
            prompt,
            model=payload.model,
            temporary=True,
        )
        saved_images: list[str] = []
        for index, image in enumerate(output.images[:4]):
            filename = f"{uuid.uuid4().hex}-{index}.png"
            try:
                saved_path = await save_generated_image(
                    image,
                    runtime.client,
                    GENERATED_DIR,
                    filename,
                )
                saved_images.append(f"/generated/{saved_path.name}")
            except Exception:
                if is_google_image_url(image.url):
                    saved_images.append(image.url)
                else:
                    raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}") from exc

    if not saved_images:
        raise HTTPException(
            status_code=422,
            detail=output.text or "Gemini did not return a generated image.",
        )

    return {"text": output.text, "images": saved_images}


@app.get("/")
async def index() -> FileResponse:
    """Serve the application shell."""
    return FileResponse(WEB_DIR / "index.html")


app.mount("/generated", StaticFiles(directory=GENERATED_DIR, check_dir=False), name="generated")
app.mount("/assets", StaticFiles(directory=ROOT_DIR / "assets"), name="assets")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    """Run the local-only development server."""
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))
    uvicorn.run("web_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
