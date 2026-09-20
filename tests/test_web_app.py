import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import web_app


class FakeModel:
    model_name = "gemini-pro"
    display_name = "Gemini Pro"
    is_available = True


class FakeImage:
    url = "https://lh3.googleusercontent.com/image.png"

    async def save(self, path: str, filename: str) -> str:
        target = Path(path) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"image")
        return str(target)


class FakeOutput:
    def __init__(self) -> None:
        self.text = "hello"
        self.images = [FakeImage()]


class FakeChat:
    send_message = AsyncMock(return_value=FakeOutput())


class FakeClient:
    account_status = SimpleNamespace(name="AVAILABLE")
    generate_content = AsyncMock(return_value=FakeOutput())

    def list_models(self) -> list[FakeModel]:
        return [FakeModel()]

    def start_chat(self, model: str) -> FakeChat:
        return FakeChat()


def fake_request(runtime: web_app.WebRuntime) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime)))


class WebAppTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.runtime = web_app.WebRuntime(self.client)  # type: ignore[arg-type]
        self.request = fake_request(self.runtime)

    async def test_status_hides_credentials_and_lists_models(self) -> None:
        result = await web_app.status(self.request)  # type: ignore[arg-type]
        assert result["ready"]
        assert result["models"][0]["name"] == "gemini-pro"
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
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(web_app, "GENERATED_DIR", Path(temp_dir)):
                result = await web_app.generate_image(
                    web_app.ImageRequest(prompt="a blue bird"),
                    self.request,  # type: ignore[arg-type]
                )
            assert len(result["images"]) == 1
            assert (Path(temp_dir) / Path(result["images"][0]).name).exists()

    async def test_image_generation_requires_an_image(self) -> None:
        self.client.generate_content = AsyncMock(
            return_value=SimpleNamespace(text="not available", images=[])
        )
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


if __name__ == "__main__":
    unittest.main()
