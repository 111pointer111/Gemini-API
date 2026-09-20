"""Local Gemini media studio with chat, queued generation, and asset downloads."""

from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import os
import uuid
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import uvicorn
from curl_cffi import CurlFollow, CurlHttpVersion
from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from gemini_webapi import ChatSession, GeminiClient, GeneratedImage, Image
from gemini_webapi.utils.browser_session import load_full_browser_cookie_session

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
GENERATED_DIR = WEB_DIR / "generated"
DEFAULT_MODEL = "gemini-pro"
MAX_IMAGE_BATCH = 4
GenerationKind = Literal["image", "video", "audio"]


class ChatRequest(BaseModel):
    """Incoming chat request from the browser."""

    message: Annotated[str, Field(min_length=1, max_length=12_000)]
    session_id: str | None = None
    model: str = DEFAULT_MODEL


class ImageRequest(BaseModel):
    """Legacy synchronous image-generation request."""

    prompt: Annotated[str, Field(min_length=1, max_length=4_000)]
    model: str = DEFAULT_MODEL


class GenerationRequest(BaseModel):
    """A queued image, video, or audio generation request."""

    kind: GenerationKind
    prompt: Annotated[str, Field(min_length=1, max_length=4_000)]
    model: str = DEFAULT_MODEL
    count: Annotated[int, Field(ge=1, le=MAX_IMAGE_BATCH)] = 1
    aspect_ratio: Literal["auto", "1:1", "16:9", "9:16", "4:3", "3:4"] = "auto"

    @model_validator(mode="after")
    def limit_non_image_batch(self) -> GenerationRequest:
        """Video and audio are long-running single-result jobs."""
        if self.kind != "image" and self.count != 1:
            raise ValueError("Video and audio jobs support one result at a time.")
        return self


class AssetSelection(BaseModel):
    """Assets selected for one ZIP download."""

    asset_ids: Annotated[list[str], Field(min_length=1, max_length=50)]


@dataclass(slots=True)
class SessionState:
    """A Gemini conversation and the lock protecting its turn order."""

    chat: ChatSession
    model: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class AssetRecord:
    """One locally persisted generated file."""

    id: str
    kind: GenerationKind
    filename: str
    prompt: str
    model: str
    created_at: str
    mime_type: str
    preview_filename: str | None = None

    def public(self) -> dict[str, Any]:
        """Return browser-safe metadata without exposing local paths."""
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "model": self.model,
            "created_at": self.created_at,
            "mime_type": self.mime_type,
            "url": f"/generated/{self.filename}",
            "preview_url": (
                f"/generated/{self.preview_filename}" if self.preview_filename else None
            ),
            "download_url": f"/api/assets/{self.id}/download",
        }


@dataclass(slots=True)
class GenerationJob:
    """In-memory generation state polled by the browser."""

    id: str
    kind: GenerationKind
    prompt: str
    model: str
    count: int
    aspect_ratio: str
    status: str = "queued"
    asset_ids: list[str] = field(default_factory=list)
    completed: int = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def public(self, assets: dict[str, AssetRecord]) -> dict[str, Any]:
        """Return the current job snapshot."""
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "model": self.model,
            "count": self.count,
            "aspect_ratio": self.aspect_ratio,
            "status": self.status,
            "completed": self.completed,
            "error": self.error,
            "created_at": self.created_at,
            "assets": [assets[item].public() for item in self.asset_ids if item in assets],
        }


class WebRuntime:
    """Mutable local state shared by the FastAPI routes."""

    def __init__(self, client: GeminiClient, generated_dir: Path = GENERATED_DIR) -> None:
        self.client = client
        self.generated_dir = generated_dir
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.generated_dir / "assets.json"
        self.sessions: dict[str, SessionState] = {}
        self.jobs: dict[str, GenerationJob] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.assets = self._load_assets()
        self._asset_lock = asyncio.Lock()
        self._generation_slots = asyncio.Semaphore(2)

    def get_session(self, session_id: str | None, model: str) -> tuple[str, SessionState]:
        """Return an existing conversation or create a new one."""
        if session_id and (state := self.sessions.get(session_id)) and state.model == model:
            return session_id, state

        new_id = uuid.uuid4().hex
        state = SessionState(chat=self.client.start_chat(model=model), model=model)
        self.sessions[new_id] = state
        return new_id, state

    def submit_generation(self, payload: GenerationRequest) -> GenerationJob:
        """Create and start a generation job without blocking its HTTP request."""
        job = GenerationJob(
            id=uuid.uuid4().hex,
            kind=payload.kind,
            prompt=payload.prompt,
            model=payload.model,
            count=payload.count,
            aspect_ratio=payload.aspect_ratio,
        )
        self.jobs[job.id] = job
        self.tasks[job.id] = asyncio.create_task(self._run_generation(job))
        return job

    async def close(self) -> None:
        """Cancel unfinished local tasks before shutting down."""
        pending = [task for task in self.tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def list_assets(self, kind: GenerationKind | None = None) -> list[dict[str, Any]]:
        """Return newest-first browser-safe asset records."""
        records = [asset for asset in self.assets.values() if kind is None or asset.kind == kind]
        records.sort(key=lambda asset: asset.created_at, reverse=True)
        return [asset.public() for asset in records]

    def asset_path(self, asset: AssetRecord) -> Path:
        """Resolve one manifest file while preventing path traversal."""
        root = self.generated_dir.resolve()
        candidate = (root / asset.filename).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise FileNotFoundError(asset.filename)
        return candidate

    def _load_assets(self) -> dict[str, AssetRecord]:
        """Load the ignored local asset manifest and drop missing files."""
        if not self.manifest_path.is_file():
            return {}
        try:
            rows = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            assets = {
                row["id"]: AssetRecord(**row)
                for row in rows
                if (self.generated_dir / row.get("filename", "")).is_file()
            }
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return {}
        return assets

    async def _persist_assets(self) -> None:
        """Atomically write the local asset manifest outside the event loop."""
        rows = [asdict(asset) for asset in self.assets.values()]
        payload = json.dumps(rows, ensure_ascii=False, indent=2)
        target = self.manifest_path
        temporary = target.with_suffix(".tmp")

        def write() -> None:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(target)

        await asyncio.to_thread(write)

    async def _register_asset(
        self,
        *,
        kind: GenerationKind,
        path: Path,
        prompt: str,
        model: str,
        preview_path: Path | None = None,
    ) -> AssetRecord:
        """Add a downloaded file to the persistent local gallery."""
        asset = AssetRecord(
            id=uuid.uuid4().hex,
            kind=kind,
            filename=path.name,
            prompt=prompt,
            model=model,
            created_at=datetime.now(UTC).isoformat(),
            mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            preview_filename=preview_path.name if preview_path and preview_path.is_file() else None,
        )
        async with self._asset_lock:
            self.assets[asset.id] = asset
            await self._persist_assets()
        return asset

    async def _run_generation(self, job: GenerationJob) -> None:
        """Execute one queued job and keep partial results visible."""
        job.status = "running"
        errors: list[str] = []
        try:
            if job.kind == "image":
                image_tasks = [
                    asyncio.create_task(self._generate_image_asset(job, index))
                    for index in range(job.count)
                ]
                try:
                    for completed_task in asyncio.as_completed(image_tasks):
                        try:
                            result = await completed_task
                        except Exception as exc:
                            errors.append(str(exc))
                            continue
                        job.asset_ids.append(result.id)
                        job.completed += 1
                finally:
                    unfinished = [task for task in image_tasks if not task.done()]
                    for task in unfinished:
                        task.cancel()
                    if unfinished:
                        await asyncio.gather(*unfinished, return_exceptions=True)
            else:
                asset = await self._generate_timed_media(job)
                job.asset_ids.append(asset.id)
                job.completed = 1
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:
            errors.append(str(exc))

        if errors:
            job.error = " · ".join(errors)[:800]
        if job.completed == job.count:
            job.status = "completed"
        elif job.completed:
            job.status = "partial"
        else:
            job.status = "failed"

    async def _generate_image_asset(self, job: GenerationJob, index: int) -> AssetRecord:
        """Generate and persist one original-size image."""
        aspect = "" if job.aspect_ratio == "auto" else f" Use a {job.aspect_ratio} aspect ratio."
        prompt = (
            "Generate exactly one original image from this description. "
            f"Do not search the web.{aspect} Description: {job.prompt}"
        )
        async with self._generation_slots:
            output = await self.client.generate_content(
                prompt,
                model=job.model,
                temporary=True,
            )
            if not output.images:
                raise ValueError(output.text or f"Image {index + 1} was not returned.")
            filename = f"{job.id}-{index + 1}.png"
            saved_path = await save_generated_image(
                output.images[0],
                self.client,
                self.generated_dir,
                filename,
            )
        return await self._register_asset(
            kind="image",
            path=saved_path,
            prompt=job.prompt,
            model=job.model,
        )

    async def _generate_timed_media(self, job: GenerationJob) -> AssetRecord:
        """Generate and persist one video or audio result."""
        instruction = (
            "Generate a video from this description: "
            if job.kind == "video"
            else "Generate an audio or music track from this description: "
        )
        async with self._generation_slots:
            output = await self.client.generate_content(
                instruction + job.prompt,
                model=job.model,
                temporary=True,
            )
            candidates = output.videos if job.kind == "video" else output.media
            if not candidates:
                raise ValueError(output.text or f"Gemini did not return {job.kind} media.")
            media = candidates[0]
            async with media_download_session(self.client) as media_client:
                if job.kind == "audio":
                    paths = await media.save(
                        path=str(self.generated_dir),
                        filename=job.id,
                        client=media_client,
                        download_type="audio",
                    )
                else:
                    paths = await media.save(
                        path=str(self.generated_dir),
                        filename=job.id,
                        client=media_client,
                    )

        primary_key = "video" if job.kind == "video" else "audio"
        primary_value = paths.get(primary_key)
        if not primary_value or primary_value == "206":
            raise ValueError(f"Gemini did not finish the {job.kind} download.")
        preview_value = paths.get("video_thumbnail") or paths.get("audio_thumbnail")
        return await self._register_asset(
            kind=job.kind,
            path=Path(primary_value),
            prompt=job.prompt,
            model=job.model,
            preview_path=Path(preview_value) if preview_value else None,
        )


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


@asynccontextmanager
async def media_download_session(client: GeminiClient) -> AsyncIterator[AsyncSession]:
    """Create an authenticated media client that uses the configured proxy."""
    async with AsyncSession(
        impersonate=client.impersonate,
        cookies=client.cookies,
        proxy=configured_proxy(),
        allow_redirects=CurlFollow.ALL,
        http_version=CurlHttpVersion.NONE,
    ) as media_client:
        yield media_client


async def save_generated_image(
    image: Image,
    client: GeminiClient,
    destination: Path,
    filename: str,
) -> Path:
    """Save one image at original size through the authenticated proxy session."""
    if not is_google_image_url(image.url):
        raise ValueError("Gemini returned an unsupported image host.")

    if not isinstance(image, GeneratedImage):
        return Path(await image.save(path=str(destination), filename=filename))

    async with media_download_session(client) as media_client:
        return Path(
            await image.save(
                path=str(destination),
                filename=filename,
                client=media_client,
                full_size=True,
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
    runtime = WebRuntime(client)
    app.state.runtime = runtime
    app.state.cookie_browser = browser_session.browser if browser_session else None
    try:
        yield
    finally:
        await runtime.close()
        await client.close()


app = FastAPI(title="Gemini Atelier", version="2.0.0", lifespan=lifespan)


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


@app.post("/api/generations", status_code=202)
async def create_generation(payload: GenerationRequest, request: Request) -> dict[str, Any]:
    """Queue a media generation and immediately return its job snapshot."""
    runtime = runtime_from(request)
    job = runtime.submit_generation(payload)
    return job.public(runtime.assets)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """Return one generation job for browser polling."""
    runtime = runtime_from(request)
    job = runtime.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return job.public(runtime.assets)


@app.get("/api/assets")
async def list_assets(
    request: Request,
    kind: Annotated[GenerationKind | None, Query()] = None,
) -> dict[str, Any]:
    """List persisted local media for the gallery."""
    runtime = runtime_from(request)
    return {"assets": runtime.list_assets(kind)}


@app.get("/api/assets/{asset_id}/download")
async def download_asset(asset_id: str, request: Request) -> FileResponse:
    """Download the locally saved original file."""
    runtime = runtime_from(request)
    asset = runtime.assets.get(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    try:
        path = runtime.asset_path(asset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Asset file is missing.") from exc
    return FileResponse(
        path,
        media_type=asset.mime_type,
        filename=f"gemini-{asset.kind}-{asset.id[:8]}{path.suffix}",
    )


@app.post("/api/assets/download-zip")
async def download_assets_zip(payload: AssetSelection, request: Request) -> Response:
    """Create a ZIP archive containing selected original files."""
    runtime = runtime_from(request)
    selected = [
        runtime.assets[item] for item in dict.fromkeys(payload.asset_ids) if item in runtime.assets
    ]
    if not selected:
        raise HTTPException(status_code=404, detail="No downloadable assets were selected.")

    def build_archive() -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, asset in enumerate(selected, start=1):
                path = runtime.asset_path(asset)
                archive.write(path, f"{index:02d}-gemini-{asset.kind}-{asset.id[:8]}{path.suffix}")
        return buffer.getvalue()

    try:
        content = await asyncio.to_thread(build_archive)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="One selected asset file is missing.") from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="gemini-assets.zip"'},
    )


@app.post("/api/images")
async def generate_image(payload: ImageRequest, request: Request) -> dict[str, Any]:
    """Keep the original synchronous image endpoint for existing clients."""
    runtime = runtime_from(request)
    prompt = f"Generate an image from this description. Do not search the web: {payload.prompt}"
    try:
        output = await runtime.client.generate_content(
            prompt,
            model=payload.model,
            temporary=True,
        )
        saved_images: list[str] = []
        for index, image in enumerate(output.images[:MAX_IMAGE_BATCH]):
            filename = f"{uuid.uuid4().hex}-{index}.png"
            saved_path = await save_generated_image(
                image,
                runtime.client,
                runtime.generated_dir,
                filename,
            )
            saved_images.append(f"/generated/{saved_path.name}")
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
