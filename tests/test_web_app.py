import tempfile
import unittest
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from curl_cffi.requests import Cookies
from fastapi import HTTPException
from pydantic import ValidationError

import web_app
from gemini_webapi import GeneratedImage
from gemini_webapi.constants import AccountStatus
from gemini_webapi.utils.cookie_config import cookies_from_mapping


class FakeModel:
    model_name = "gemini-pro"
    display_name = "Gemini Pro"
    is_available = True


class FakeImage:
    url = "https://lh3.googleusercontent.com/image.png"

    async def save(self, path: str, filename: str) -> str:
        if not Path(filename).suffix:
            filename = f"{filename}.jpg"
        target = Path(path) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"image")
        return str(target)


class FakeVideo:
    async def save(
        self,
        path: str,
        filename: str,
        client: object,
        **kwargs: str,
    ) -> dict[str, str | None]:
        suffix = ".mp3" if kwargs.get("download_type") == "audio" else ".mp4"
        target = Path(path) / f"{filename}{suffix}"
        target.write_bytes(b"media")
        return {
            "audio": str(target) if suffix == ".mp3" else None,
            "video": str(target) if suffix == ".mp4" else None,
            "video_thumbnail": None,
            "audio_thumbnail": None,
        }


class FakeOutput:
    def __init__(
        self,
        *,
        images: list[FakeImage] | None = None,
        videos: list[FakeVideo] | None = None,
        media: list[FakeVideo] | None = None,
    ) -> None:
        self.text = "hello"
        self.images = images if images is not None else [FakeImage()]
        self.videos = videos or []
        self.media = media or []


class FakeChat:
    def __init__(self) -> None:
        self.send_message = AsyncMock(return_value=FakeOutput())


class FakeClient:
    account_status = AccountStatus.AVAILABLE
    impersonate = "chrome"

    def __init__(self) -> None:
        self.cookies: Cookies | dict[str, str] = {}
        self.generate_content = AsyncMock(return_value=FakeOutput())
        self.close = AsyncMock()

    def list_models(self) -> list[FakeModel]:
        return [FakeModel()]

    def start_chat(self, model: str) -> FakeChat:
        return FakeChat()


def fake_request(runtime: web_app.WebRuntime) -> SimpleNamespace:
    state = SimpleNamespace(runtime=runtime)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class WebAppTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.client = FakeClient()
        self.runtime = web_app.WebRuntime(
            self.client,  # pyright: ignore[reportArgumentType]
            generated_dir=Path(self.temp_dir.name),
            cookie_config_path=Path(self.temp_dir.name) / ".gemini-cookie-session.json",
        )
        self.request = fake_request(self.runtime)

    async def asyncTearDown(self) -> None:
        await self.runtime.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_status_hides_credentials_and_lists_models(self) -> None:
        result = await web_app.status(self.request)  # type: ignore[arg-type]
        assert result["ready"]
        assert result["models"][0]["name"] == "gemini-pro"
        assert result["cookie_source"] == "browser"
        assert not result["saved_cookie_config"]
        assert "cookies" not in result

    async def test_chat_reuses_session(self) -> None:
        first = await web_app.chat(
            web_app.ChatRequest(message="hello", model="gemini-pro"),
            self.request,  # type: ignore[arg-type]
        )
        second = await web_app.chat(
            web_app.ChatRequest(
                message="again", session_id=first["session_id"], model="gemini-pro"
            ),
            self.request,  # type: ignore[arg-type]
        )
        assert first["session_id"] == second["session_id"]
        assert second["text"] == "hello"

    async def test_image_generation_saves_local_file(self) -> None:
        result = await web_app.generate_image(
            web_app.ImageRequest(prompt="a blue bird"),
            self.request,  # type: ignore[arg-type]
        )
        assert len(result["images"]) == 1
        assert (self.runtime.generated_dir / Path(result["images"][0]).name).exists()

    async def test_image_generation_requires_an_image(self) -> None:
        self.client.generate_content = AsyncMock(return_value=FakeOutput(images=[]))
        caught: HTTPException | None = None
        try:
            await web_app.generate_image(
                web_app.ImageRequest(prompt="a blue bird"),
                self.request,  # type: ignore[arg-type]
            )
        except HTTPException as exc:
            caught = exc
        assert caught is not None
        assert caught.status_code == 422

    async def test_batch_job_generates_independent_original_assets(self) -> None:
        job = self.runtime.submit_generation(
            web_app.GenerationRequest(
                kind="image",
                prompt="three blue birds",
                count=3,
                aspect_ratio="1:1",
            )
        )
        await self.runtime.tasks[job.id]

        assert job.status == "completed"
        assert job.completed == 3
        assert len(job.asset_ids) == 3
        assert self.client.generate_content.await_count == 3
        assets = self.runtime.list_assets("image")
        assert len(assets) == 3
        assert all(asset["source_url"].startswith("https://") for asset in assets)
        assert all(asset["download_url"].endswith("/original") for asset in assets)
        assert all(asset["local_download_url"].startswith("/api/assets/") for asset in assets)
        assert all(asset["url"].endswith(".jpg") for asset in assets)
        assert self.runtime.manifest_path.is_file()

    async def test_original_image_download_is_proxied_with_correct_type(self) -> None:
        local_file = self.runtime.generated_dir / "preview.jpg"
        local_file.write_bytes(b"local-preview")
        asset = await self.runtime._register_asset(
            kind="image",
            path=local_file,
            prompt="original",
            model="gemini-pro",
            source_url="https://lh3.googleusercontent.com/full-size-image",
        )
        original_bytes = b"\xff\xd8\xff\xe0original-jpeg-bytes"
        upstream_response = SimpleNamespace(
            content=original_bytes,
            headers={"content-type": "image/jpeg; charset=binary"},
            raise_for_status=lambda: None,
        )
        media_client = SimpleNamespace(get=AsyncMock(return_value=upstream_response))

        @asynccontextmanager
        async def fake_media_session(client: object):
            assert client is self.client
            yield media_client

        with patch.object(web_app, "media_download_session", fake_media_session):
            response = await web_app.download_original_asset(
                asset.id,
                self.request,  # type: ignore[arg-type]
            )

        assert response.body == original_bytes
        assert response.media_type == "image/jpeg"
        assert response.headers["content-disposition"].endswith('.jpg"')
        media_client.get.assert_awaited_once_with(
            asset.source_url,
            headers=web_app.Headers.REFERER.value,
        )

    async def test_unauthenticated_session_blocks_generation(self) -> None:
        self.client.account_status = AccountStatus.UNAUTHENTICATED
        caught: HTTPException | None = None
        try:
            await web_app.create_generation(
                web_app.GenerationRequest(kind="image", prompt="blocked"),
                self.request,  # type: ignore[arg-type]
            )
        except HTTPException as exc:
            caught = exc
        assert caught is not None
        assert caught.status_code == 401
        assert not self.runtime.jobs

    async def test_cookie_configuration_validates_persists_and_hot_swaps(self) -> None:
        cookie_values = {
            "__Secure-1PSID": "session-value",
            "__Secure-1PSIDTS": "timestamp-value",
            "NID": "extra-value",
        }
        candidate = FakeClient()
        candidate.cookies = cookies_from_mapping(cookie_values)

        with (
            patch.object(web_app, "configured_proxy", return_value="http://proxy.test:7890"),
            patch.object(
                web_app,
                "initialize_gemini_client",
                new=AsyncMock(return_value=candidate),
            ) as initialize,
        ):
            result = await web_app.configure_cookies(
                web_app.CookieConfigRequest(
                    cookie_header=(
                        "__Secure-1PSID=session-value; "
                        "__Secure-1PSIDTS=timestamp-value; NID=extra-value"
                    )
                ),
                self.request,  # type: ignore[arg-type]
            )

        assert result["ready"]
        assert result["cookie_source"] == "manual"
        assert result["saved_cookie_config"]
        assert self.runtime.client is candidate
        assert self.client.close.await_count == 1
        initialize.assert_awaited_once()
        assert self.runtime.cookie_config_path.stat().st_mode & 0o777 == 0o600

    async def test_video_and_audio_jobs_save_the_expected_media_type(self) -> None:
        self.client.generate_content = AsyncMock(
            side_effect=[
                FakeOutput(images=[], videos=[FakeVideo()]),
                FakeOutput(images=[], media=[FakeVideo()]),
            ]
        )
        video = self.runtime.submit_generation(
            web_app.GenerationRequest(kind="video", prompt="waves", count=1)
        )
        audio = self.runtime.submit_generation(
            web_app.GenerationRequest(kind="audio", prompt="soft synth", count=1)
        )
        await self.runtime.tasks[video.id]
        await self.runtime.tasks[audio.id]

        assert video.status == "completed"
        assert audio.status == "completed"
        assert self.runtime.assets[video.asset_ids[0]].filename.endswith(".mp4")
        assert self.runtime.assets[audio.asset_ids[0]].filename.endswith(".mp3")

    async def test_original_image_flag_is_enabled(self) -> None:
        image = GeneratedImage(url="https://lh3.googleusercontent.com/image.png")
        destination = self.runtime.generated_dir / "original.png"
        with patch.object(
            GeneratedImage,
            "save",
            new=AsyncMock(return_value=str(destination)),
        ) as save:
            result = await web_app.save_generated_image(
                image,
                self.client,  # type: ignore[arg-type]
                self.runtime.generated_dir,
                destination.name,
            )
        assert result == destination
        assert save.await_args is not None
        assert save.await_args.kwargs["full_size"] is True

    async def test_zip_download_contains_selected_originals(self) -> None:
        first = self.runtime.generated_dir / "first.png"
        second = self.runtime.generated_dir / "second.png"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        asset_one = await self.runtime._register_asset(
            kind="image", path=first, prompt="one", model="gemini-pro"
        )
        asset_two = await self.runtime._register_asset(
            kind="image", path=second, prompt="two", model="gemini-pro"
        )

        response = await web_app.download_assets_zip(
            web_app.AssetSelection(asset_ids=[asset_one.id, asset_two.id]),
            self.request,  # type: ignore[arg-type]
        )
        with zipfile.ZipFile(BytesIO(response.body)) as archive:
            assert len(archive.namelist()) == 2
            assert sorted(archive.read(name) for name in archive.namelist()) == [b"one", b"two"]

    def test_video_batch_is_rejected(self) -> None:
        caught: ValidationError | None = None
        try:
            web_app.GenerationRequest(kind="video", prompt="waves", count=2)
        except ValidationError as exc:
            caught = exc
        assert caught is not None


if __name__ == "__main__":
    unittest.main()
