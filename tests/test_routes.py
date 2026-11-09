"""Tests for app/routes.py — endpoint integration tests."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.errors import AppError
from app.main import register_error_handlers
from app.routes import router


@pytest.fixture
def test_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(router)
    app.state.models_loaded = True
    app.state.embedding_model = None
    app.state.gemini_client = None
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


@pytest.fixture
def sample_pdf_bytes():
    return b"%PDF-1.4 fake pdf content for testing"


class TestHealthEndpoint:
    def test_health_returns_envelope(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "code" in body
        assert "message" in body


class TestExtractEndpoint:
    def test_extract_validation_no_files(self, client):
        response = client.post("/extract")
        assert response.status_code == 400
        body = response.json()
        assert body["code"] == 500040

    def test_extract_success_single_file(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_file_with_files_api.return_value = {
            "raw_text": "hello world", "ids": [],
        }
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        with (
            patch.object(client.app.state, "gemini_client", mock_gc),
            patch.object(client.app.state, "embedding_model", mock_model),
        ):
            response = client.post(
                "/extract",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500100
        assert len(body["data"]) == 1
        assert body["data"][0]["text"] == "hello world"

    def test_extract_failure_per_file(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_file_with_files_api.side_effect = RuntimeError("Gemini failed")

        with patch.object(client.app.state, "gemini_client", mock_gc):
            response = client.post(
                "/extract",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500150


    def test_extract_duplicate_filenames_processed_independently(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_file_with_files_api.side_effect = [
            {"raw_text": "doc 1", "ids": []},
            {"raw_text": "doc 2", "ids": []},
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        with (
            patch.object(client.app.state, "gemini_client", mock_gc),
            patch.object(client.app.state, "embedding_model", mock_model),
        ):
            response = client.post(
                "/extract",
                files=[
                    ("files", ("same_name.pdf", sample_pdf_bytes, "application/pdf")),
                    ("files", ("same_name.pdf", sample_pdf_bytes, "application/pdf")),
                ],
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500100
        assert len(body["data"]) == 2
        assert body["data"][0]["text"] == "doc 1"
        assert body["data"][1]["text"] == "doc 2"

    def test_extract_partial_batch_failure_isolates_sibling(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_file_with_files_api.side_effect = [
            {"raw_text": "doc 1", "ids": []},
            Exception("Gemini upload failed"),
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        with (
            patch.object(client.app.state, "gemini_client", mock_gc),
            patch.object(client.app.state, "embedding_model", mock_model),
        ):
            response = client.post(
                "/extract",
                files=[
                    ("files", ("file1.pdf", sample_pdf_bytes, "application/pdf")),
                    ("files", ("file2.pdf", sample_pdf_bytes, "application/pdf")),
                ],
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500100
        assert len(body["data"]) == 2
        assert body["data"][0]["text"] == "doc 1"
        assert body["data"][1] is None
        assert "files.1" in body["errors"]
        assert "files.0" not in body.get("errors", {})


class TestVerifyEndpoint:
    def test_verify_success(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_direct.return_value = {
            "raw_text": "doc text", "ids": []
        }
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        compared = json.dumps([[0.1, 0.2, 0.3]])

        with (
            patch.object(client.app.state, "gemini_client", mock_gc),
            patch.object(client.app.state, "embedding_model", mock_model),
        ):
            response = client.post(
                "/verify",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
                data={"embeddings": compared},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500200
        assert len(body["data"]) == 1
        assert "similarity_score" in body["data"][0]
        assert "verdict" in body["data"][0]
        assert "descriptions" in body["data"][0]

    @pytest.mark.parametrize("failure", [
        RuntimeError("503 UNAVAILABLE"),
        AppError(500150, "Invalid JSON response from Gemini"),
        AppError(500050, "Embedding failed"),
    ])
    def test_verify_technical_failure_is_not_document_error(
        self, client, sample_pdf_bytes, failure,
    ):
        mock_gc = MagicMock()
        mock_gc.extract_direct.side_effect = failure

        with patch.object(client.app.state, "gemini_client", mock_gc):
            response = client.post(
                "/verify",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
                data={"embeddings": "[[0.1, 0.2, 0.3]]"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500150
        assert body["data"] == [None]
        assert "files.0" in body["errors"]

    def test_verify_empty_text_remains_document_error(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_direct.return_value = {"raw_text": "", "ids": []}

        with patch.object(client.app.state, "gemini_client", mock_gc):
            response = client.post(
                "/verify",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
                data={"embeddings": "[[0.1, 0.2, 0.3]]"},
            )

        assert response.status_code == 200
        assert response.json()["code"] == 500240
        assert response.json()["data"] == [None]

    def test_verify_metadata_mismatch(self, client, sample_pdf_bytes):
        compared = json.dumps([[0.1], [0.2]])

        response = client.post(
            "/verify",
            files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            data={"embeddings": compared},
        )

        assert response.status_code == 400
        body = response.json()
        assert body["code"] == 500241

    def test_parse_embeddings_rejects_nan_and_infinity(self, client, sample_pdf_bytes):
        response = client.post(
            "/verify",
            files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            data={"embeddings": "[[NaN]]"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == 500241

        response2 = client.post(
            "/verify",
            files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            data={"embeddings": "[[Infinity]]"},
        )
        assert response2.status_code == 400
        assert response2.json()["code"] == 500241


class TestExtractIdsEndpoint:
    def test_extract_ids_success(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_ids_direct.return_value = [
            {"type": "nik", "value": "1234567890123456"}
        ]

        with patch.object(client.app.state, "gemini_client", mock_gc):
            response = client.post(
                "/extract-ids",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500300
        assert body["data"][0]["ids"] == [
            {"type": "nik", "value": "1234567890123456"}
        ]

    def test_extract_ids_genuine_empty_list_returns_500300(self, client, sample_pdf_bytes):
        mock_gc = MagicMock()
        mock_gc.extract_ids_direct.return_value = []

        with patch.object(client.app.state, "gemini_client", mock_gc):
            response = client.post(
                "/extract-ids",
                files=[("files", ("test.pdf", sample_pdf_bytes, "application/pdf"))],
            )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 500300
        assert body["data"][0]["ids"] == []
