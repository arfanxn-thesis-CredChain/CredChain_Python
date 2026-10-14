"""Tests for app/gemini.py — Gemini client for document extraction."""

import json
from unittest.mock import MagicMock

import pytest

from app.gemini import GeminiClient


@pytest.fixture
def mock_genai_client():
    return MagicMock()


class TestGeminiClient:
    def test_extract_document_returns_parsed_json(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_response = MagicMock()
        mock_response.text = json.dumps({"raw_text": "Hello", "ids": []})
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client._extract_document(["test content"])

        assert result == {"raw_text": "Hello", "ids": []}

    def test_extract_document_empty_response_returns_empty_dict(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_response = MagicMock()
        mock_response.text = None
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client._extract_document(["test content"])

        assert result == {}

    def test_extract_document_invalid_json_returns_empty_dict(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_response = MagicMock()
        mock_response.text = "not json"
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client._extract_document(["test content"])

        assert result == {}

    def test_extract_document_with_retry_succeeds(self, mock_genai_client):
        client = GeminiClient(
            mock_genai_client, extraction_model="gemini-test", retry_wait_seconds=0
        )
        mock_response = MagicMock()
        mock_response.text = json.dumps({"raw_text": "OK"})
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client._extract_document_with_retry(["test content"])

        assert result == {"raw_text": "OK"}

    def test_extract_document_with_retry_handles_rate_limit(self, mock_genai_client):
        client = GeminiClient(
            mock_genai_client, extraction_model="gemini-test", retry_wait_seconds=0
        )
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("429 RESOURCE_EXHAUSTED")
            mock_response = MagicMock()
            mock_response.text = json.dumps({"raw_text": "retry ok"})
            return mock_response

        mock_genai_client.models.generate_content.side_effect = side_effect

        result = client._extract_document_with_retry(["test content"])

        assert result == {"raw_text": "retry ok"}
        assert call_count[0] == 3

    def test_extract_document_with_retry_exhausts(self, mock_genai_client):
        client = GeminiClient(
            mock_genai_client, extraction_model="gemini-test", retry_wait_seconds=0,
            max_retries=2,
        )
        mock_genai_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED"
        )

        with pytest.raises(RuntimeError, match="Max retries exceeded"):
            client._extract_document_with_retry(["test content"])

    def test_extract_direct_single_file(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {"raw_text": "doc text", "ids": [{"type": "passport", "value": "X123"}]}
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client.extract_direct(
            file_bytes=b"fake pdf bytes",
            mime_type="application/pdf",
            prompt="extract all",
        )

        assert result == {
            "raw_text": "doc text",
            "ids": [{"type": "passport", "value": "X123"}],
        }

    def test_extract_ids_direct(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {"ids": [{"type": "nik", "value": "1234567890123456"}]}
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client.extract_ids_direct(
            file_bytes=b"fake image", mime_type="image/jpeg", prompt="extract ids"
        )

        assert result == [{"type": "nik", "value": "1234567890123456"}]

    def test_extract_ids_direct_no_ids(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_response = MagicMock()
        mock_response.text = json.dumps({"ids": []})
        mock_genai_client.models.generate_content.return_value = mock_response

        result = client.extract_ids_direct(
            file_bytes=b"fake image", mime_type="image/jpeg", prompt="extract ids"
        )

        assert result == []

    def test_upload_file(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_file = MagicMock()
        mock_file.state = "PROCESSING"
        mock_genai_client.files.upload.return_value = mock_file

        result = client.upload_file(
            file_bytes=b"test", mime_type="application/pdf", display_name="doc.pdf"
        )

        assert result == mock_file
        mock_genai_client.files.upload.assert_called_once()

    def test_poll_until_active(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_file = MagicMock()
        mock_file.name = "files/abc"
        mock_file_info = MagicMock()
        mock_file_info.state = "ACTIVE"
        mock_genai_client.files.get.return_value = mock_file_info

        result = client.poll_until_active(mock_file)

        assert result.state == "ACTIVE"

    def test_poll_until_active_fails(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_file = MagicMock()
        mock_file.name = "files/abc"
        mock_file_info = MagicMock()
        mock_file_info.state = "FAILED"
        mock_genai_client.files.get.return_value = mock_file_info

        with pytest.raises(RuntimeError, match="failed processing"):
            client.poll_until_active(mock_file)

    def test_extract_with_files_api(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/abc"
        mock_uploaded_file.uri = "https://example.com/file"
        mock_uploaded_file.mime_type = "application/pdf"
        mock_genai_client.files.upload.return_value = mock_uploaded_file

        mock_file_info = MagicMock()
        mock_file_info.state = "ACTIVE"
        mock_file_info.uri = "https://example.com/file"
        mock_file_info.mime_type = "application/pdf"
        mock_genai_client.files.get.return_value = mock_file_info

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {"raw_text": "extracted", "ids": []}
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        file_dict = {"doc.pdf": b"fake pdf bytes"}
        results = client.extract_with_files_api(file_dict, "extract prompt")

        assert len(results) == 1
        name, raw_dict = results[0]
        assert name == "doc.pdf"
        assert raw_dict == {"raw_text": "extracted", "ids": []}

    @pytest.mark.parametrize("filename", ["scan.tif", "scan.TIF", "scan.tiff"])
    def test_extract_with_files_api_uploads_tiff_mime(
        self, mock_genai_client, filename, monkeypatch
    ):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/abc"
        mock_genai_client.files.upload.return_value = mock_uploaded_file
        mock_file_info = MagicMock()
        mock_file_info.state = "ACTIVE"
        mock_file_info.uri = "https://example.com/file"
        mock_file_info.mime_type = "image/tiff"
        mock_genai_client.files.get.return_value = mock_file_info
        mock_genai_client.models.generate_content.return_value.text = "{}"
        monkeypatch.setattr("app.gemini.time.sleep", lambda _: None)

        client.extract_with_files_api({filename: b"tiff bytes"}, "extract prompt")

        config = mock_genai_client.files.upload.call_args.kwargs["config"]
        assert config.mime_type == "image/tiff"
        assert config.display_name == filename

    def test_extract_with_files_api_handles_failure(self, mock_genai_client):
        client = GeminiClient(mock_genai_client, extraction_model="gemini-test")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/abc"
        mock_uploaded_file.uri = "https://example.com/file"
        mock_uploaded_file.mime_type = "application/pdf"
        mock_genai_client.files.upload.return_value = mock_uploaded_file

        mock_file_info = MagicMock()
        mock_file_info.state = "ACTIVE"
        mock_file_info.uri = "https://example.com/file"
        mock_file_info.mime_type = "application/pdf"
        mock_genai_client.files.get.return_value = mock_file_info

        mock_genai_client.models.generate_content.side_effect = RuntimeError(
            "Max retries exceeded"
        )

        file_dict = {"doc.pdf": b"fake pdf bytes"}
        results = client.extract_with_files_api(file_dict, "extract prompt")

        assert len(results) == 0
