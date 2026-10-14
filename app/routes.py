"""FastAPI router defining all four endpoints."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app import codes, schemas, verdict
from app import description as desc_module
from app import embeddings as emb_mod
from app.config import settings
from app.errors import AppError
from app.logger import get_logger
from app.prompts import PROMPT_EXTRACT_DOCUMENT, PROMPT_EXTRACT_IDS

router = APIRouter()
log = get_logger("routes")

ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "application/pdf", "image/jpeg", "image/png", "image/webp", "image/tiff",
})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def validate_file(file: UploadFile) -> tuple[bytes, str]:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise AppError(
            codes.CODE_AI_VALIDATION,
            errors={"file": [f"Unsupported MIME type: {file.content_type}"]},
        )
    contents = file.file.read()
    if not contents:
        raise AppError(codes.CODE_AI_VALIDATION, errors={"file": ["File is empty"]})
    if len(contents) > MAX_UPLOAD_BYTES:
        raise AppError(
            codes.CODE_AI_VALIDATION,
            errors={"file": [f"File too large: {len(contents)} bytes > {MAX_UPLOAD_BYTES}"]},
        )
    file.file.seek(0)
    return contents, file.content_type


def parse_embeddings(raw: str, expected_len: int) -> list[list[float]]:
    try:
        parsed: list[Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AppError(
            codes.CODE_AI_VERIFY_INVALID_INPUT,
            errors={"embeddings": [str(exc)]},
        ) from exc
    if not isinstance(parsed, list):
        raise AppError(
            codes.CODE_AI_VERIFY_INVALID_INPUT,
            errors={"embeddings": ["Must be a JSON array"]},
        )
    if len(parsed) != expected_len:
        raise AppError(
            codes.CODE_AI_VERIFY_INVALID_INPUT,
            errors={"embeddings": [
                f"Length mismatch: {len(parsed)} vs {expected_len}"
            ]},
        )
    result: list[list[float]] = []
    for i, v in enumerate(parsed):
        if not isinstance(v, list):
            raise AppError(
                codes.CODE_AI_VERIFY_INVALID_INPUT,
                errors={f"embeddings.{i}": ["Must be a list of floats"]},
            )
        result.append([float(x) for x in v])
    return result


def validate_files(files: list[UploadFile]) -> None:
    if not files:
        raise AppError(
            codes.CODE_AI_VALIDATION,
            errors={"files": ["At least one file is required"]},
        )
    if len(files) > settings.max_files_per_request:
        raise AppError(
            codes.CODE_AI_VALIDATION,
            errors={"files": [f"Too many files: {len(files)} > {settings.max_files_per_request}"]},
        )


def get_embedding_model(request: Request) -> Any:
    return request.app.state.embedding_model


def get_gemini_client(request: Request) -> Any:
    return request.app.state.gemini_client


def get_lang(request: Request) -> str:
    return getattr(request.state, "lang", "id")


@router.post("/extract", response_model=schemas.Response[list[schemas.ExtractData | None]])  # type: ignore[no-untyped-def]
async def extract(
    files: list[UploadFile] = File(...),
    gemini_client=Depends(get_gemini_client),
    embed_model=Depends(get_embedding_model),
) -> schemas.Response[list[schemas.ExtractData | None]]:
    validate_files(files)
    n = len(files)
    data: list[schemas.ExtractData | None] = [None] * n
    errors: dict[str, list[str]] = {}
    success_count = 0

    file_dict: dict[str, bytes] = {}
    file_map: dict[str, int] = {}
    for i, file in enumerate(files):
        try:
            contents, _ = validate_file(file)
            name = file.filename or f"file_{i}"
            file_dict[name] = contents
            file_map[name] = i
        except AppError as exc:
            errors[f"files.{i}"] = [exc.message]

    if file_dict:
        try:
            results = await asyncio.to_thread(
                gemini_client.extract_with_files_api, file_dict, PROMPT_EXTRACT_DOCUMENT
            )
            result_map: dict[int, tuple[str, dict[str, Any]]] = {}
            for name, raw in results:
                result_map[file_map[name]] = (name, raw)

            for i in range(n):
                if i in result_map:
                    _, raw = result_map[i]
                    raw_text = raw.get("raw_text", "")
                    ids = raw.get("ids", [])
                    emb = await asyncio.to_thread(emb_mod.encode, embed_model, raw_text)
                    data[i] = schemas.ExtractData(text=raw_text, ids=ids, embedding=emb)
                    success_count += 1
                else:
                    errors.setdefault(f"files.{i}", []).append("Gemini extraction failed")
        except Exception:
            log.exception("Gemini extraction failed")
            for i in range(n):
                if f"files.{i}" not in errors:
                    errors[f"files.{i}"] = ["Internal error processing file"]

    code = codes.CODE_AI_EXTRACT_SUCCESS if success_count > 0 else codes.CODE_AI_GEMINI_FAILED
    message = (
        "Document(s) extracted successfully"
        if success_count == n
        else f"{success_count}/{n} files extracted"
    )
    return schemas.Response(code=code, message=message, data=data, errors=errors or None)


@router.post("/verify", response_model=schemas.Response[list[schemas.VerifyData | None]])  # type: ignore[no-untyped-def]
async def verify(
    request: Request,
    files: list[UploadFile] = File(...),
    embeddings: str = Form(...),
    gemini_client=Depends(get_gemini_client),
    embed_model=Depends(get_embedding_model),
) -> schemas.Response[list[schemas.VerifyData | None]]:
    validate_files(files)
    stored = parse_embeddings(embeddings, expected_len=len(files))
    data: list[schemas.VerifyData | None] = []
    errors: dict[str, list[str]] = {}
    success_count = 0

    for i, (file, stored_embedding) in enumerate(zip(files, stored, strict=True)):
        try:
            file_bytes, mime_type = validate_file(file)
            raw = await asyncio.to_thread(
                gemini_client.extract_direct, file_bytes, mime_type, PROMPT_EXTRACT_DOCUMENT
            )
            raw_text = raw.get("raw_text", "")
            if not raw_text:
                data.append(None)
                errors[f"files.{i}"] = ["No text extracted from document"]
                continue
            embeddings_list = await asyncio.to_thread(emb_mod.encode, embed_model, raw_text)
            similarity = emb_mod.cosine_similarity(embeddings_list, stored_embedding)
            verdict_label = verdict.verdict_for(similarity)
            sim_percent = verdict.format_percent(similarity)
            descs = desc_module.build_description(verdict_label, sim_percent)
            data.append(schemas.VerifyData(
                similarity_score=similarity, similarity_percent=sim_percent,
                verdict=verdict_label, descriptions=descs,
            ))
            success_count += 1
        except AppError as exc:
            data.append(None)
            errors[f"files.{i}"] = [exc.message]
        except Exception:
            data.append(None)
            log.exception(
                "unhandled per-file error",
                extra={"extra_fields": {"file_index": i, "filename": file.filename}},
            )
            errors[f"files.{i}"] = ["Internal error processing file"]

    code = codes.CODE_AI_VERIFY_SUCCESS if success_count > 0 else codes.CODE_AI_VERIFY_OCR_FAILED
    message = (
        "Verification(s) completed"
        if success_count == len(files)
        else f"{success_count}/{len(files)} files verified"
    )
    return schemas.Response(code=code, message=message, data=data, errors=errors or None)


@router.post("/extract-ids", response_model=schemas.Response[list[schemas.ExtractIdsData | None]])  # type: ignore[no-untyped-def]
async def extract_ids(
    files: list[UploadFile] = File(...),
    gemini_client=Depends(get_gemini_client),
) -> schemas.Response[list[schemas.ExtractIdsData | None]]:
    validate_files(files)
    data: list[schemas.ExtractIdsData | None] = []
    errors: dict[str, list[str]] = {}
    success_count = 0

    for i, file in enumerate(files):
        try:
            file_bytes, mime_type = validate_file(file)
            ids = await asyncio.to_thread(
                gemini_client.extract_ids_direct, file_bytes, mime_type, PROMPT_EXTRACT_IDS
            )
            data.append(schemas.ExtractIdsData(ids=ids))
            success_count += 1
        except AppError as exc:
            data.append(None)
            errors[f"files.{i}"] = [exc.message]
        except Exception:
            data.append(None)
            log.exception(
                "unhandled per-file error",
                extra={"extra_fields": {"file_index": i, "filename": file.filename}},
            )
            errors[f"files.{i}"] = ["Internal error processing file"]

    code = (
        codes.CODE_AI_EXTRACT_IDS_SUCCESS
        if success_count > 0
        else codes.CODE_AI_EXTRACT_IDS_OCR_FAILED
    )
    message = (
        "Potential IDs extracted"
        if success_count == len(files)
        else f"{success_count}/{len(files)} files processed"
    )
    return schemas.Response(code=code, message=message, data=data, errors=errors or None)


@router.get("/health", summary="Liveness check")
async def health(request: Request) -> JSONResponse:
    models_loaded = getattr(request.app.state, "models_loaded", False)
    code = codes.CODE_AI_HEALTH_SUCCESS if models_loaded else codes.CODE_AI_HEALTH_NOT_READY
    http_status = 200 if models_loaded else 503
    message = "healthy" if models_loaded else "model loading"
    return JSONResponse(status_code=http_status, content={"code": code, "message": message})
