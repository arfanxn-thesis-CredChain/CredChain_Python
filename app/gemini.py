"""Gemini client for document extraction — Files API + direct upload.

Mirrors the pipeline from notebooks/credchain-python.ipynb:
  - /extract: Files API (upload → poll ACTIVE → prompt)
  - /verify, /extract-ids: direct upload (bytes in prompt)
"""

import io
import json
import time
from typing import Any, cast

from google import genai
from google.genai import types

MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


class GeminiClient:
    def __init__(
        self,
        client: genai.Client,
        extraction_model: str,
        retry_wait_seconds: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._extraction_model = extraction_model
        self._retry_wait_seconds = retry_wait_seconds
        self._max_retries = max_retries

    # ── Direct upload (used by /verify and /extract-ids) ────

    def extract_direct(
        self, file_bytes: bytes, mime_type: str, prompt: str
    ) -> dict[str, Any]:
        contents = [
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            types.Part.from_text(text=prompt),
        ]
        return self._extract_document_with_retry(contents)

    def extract_ids_direct(
        self, file_bytes: bytes, mime_type: str, prompt: str
    ) -> list[dict[str, str]]:
        raw = self.extract_direct(file_bytes, mime_type, prompt)
        return cast(list[dict[str, str]], raw.get("ids", []))

    # ── Files API (used by /extract for batch uploads) ──────

    def upload_file(
        self, file_bytes: bytes, mime_type: str, display_name: str,
    ) -> types.File:
        return self._client.files.upload(
            file=io.BytesIO(file_bytes),
            config=types.UploadFileConfig(
                mime_type=mime_type, display_name=display_name
            ),
        )

    def poll_until_active(self, file: types.File) -> types.File:
        while True:
            info = self._client.files.get(name=file.name or "")
            if info.state == "ACTIVE":
                return info
            if info.state == "FAILED":
                raise RuntimeError(f"File '{file.name}' failed processing")
            time.sleep(1)

    def extract_with_files_api(
        self, file_dict: dict[str, bytes], prompt: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        print(f"Uploading {len(file_dict)} file(s)...")
        uploaded = []
        for name, data in file_dict.items():
            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            mime = MIME_TYPES.get(ext, "application/octet-stream")
            f = self.upload_file(data, mime, name)
            print(f"  Uploaded '{name}' ({f.state})")
            info = self.poll_until_active(f)
            uploaded.append((name, info))

        results: list[tuple[str, dict[str, Any]]] = []
        for name, info in uploaded:
            print(f"  Extracting '{name}'...")
            contents = [
                types.Part.from_uri(
                    file_uri=info.uri or "", mime_type=info.mime_type or "",
                ),
                types.Part.from_text(text=prompt),
            ]
            try:
                raw = self._extract_document_with_retry(contents)
                results.append((name, raw))
                print("    Done")
            except RuntimeError:
                print(f"    Failed after retries, skipping '{name}'")
            time.sleep(1)

        return results

    # ── Internal helpers ────────────────────────────────────

    def _extract_document(self, contents: list[Any]) -> dict[str, Any]:
        response = self._client.models.generate_content(
            model=self._extraction_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        if not response.text:
            return {}
        try:
            return cast(dict[str, Any], json.loads(response.text))
        except json.JSONDecodeError:
            print(f"  Warning: Gemini returned non-JSON: {response.text[:200]}")
            return {}

    def _extract_document_with_retry(
        self, contents: list[Any], max_retries: int | None = None,
    ) -> dict[str, Any]:
        max_retries = max_retries or self._max_retries
        for attempt in range(max_retries):
            try:
                return self._extract_document(contents)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(
                        f"  Rate limited — retrying in {self._retry_wait_seconds}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(self._retry_wait_seconds)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")
