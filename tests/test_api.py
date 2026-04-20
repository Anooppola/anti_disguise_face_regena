"""
tests/test_api.py - Integration tests for the FastAPI backend
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── Patch InferenceEngine before app import ────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    # Create a mock engine that returns a blank white PNG
    mock_engine = MagicMock()

    def fake_predict_bytes(data, fmt="PNG"):
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    mock_engine.predict_bytes.side_effect = fake_predict_bytes

    with patch("api.app.engine", mock_engine):
        from api.app import app
        with TestClient(app) as c:
            yield c


def make_image_bytes(color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (256, 256), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoints:
    def test_root_ok(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_info_ok(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        assert "architecture" in r.json()


class TestPredictEndpoint:
    def test_predict_success(self, client):
        img_bytes = make_image_bytes()
        r = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        # Output should be a valid PNG image
        result = Image.open(io.BytesIO(r.content))
        assert result.size == (256, 256)

    def test_predict_invalid_content_type(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert r.status_code == 400
