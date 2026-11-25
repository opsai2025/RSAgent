"""
One small function for every LLM call in the agent: `chat_json()`.

It uses the `openai` SDK directly against your OpenAI-compatible endpoint
(OPENAI_API_KEY + OPENAI_BASE_URL in .env) and owns the request body, which the
reflection_scenario project showed matters on this proxy: a parameter we do not
want sent must be ABSENT, not defaulted.

Contract: the JSON schema is written into the system prompt, the reply is parsed
from plain text and validated with Pydantic. That works on every model, so there
is no ladder of request shapes. `json_mode=True` additionally asks for
response_format=json_object and silently drops it if the model rejects it.
Transient failures (timeout, connection, 429, 5xx) are retried; 4xx are not.
"""

import base64
import io
import json
import math
import os
import re
import time
from typing import List, Optional, Tuple, Type

from PIL import Image

from pydantic import BaseModel, ValidationError

import config

_client = None


class LLMError(RuntimeError):
    pass


class ContextWindowError(LLMError):
    """The caller must reduce its payload; retrying it unchanged cannot help."""


def is_context_window_error(exc):
    text = str(exc).lower()
    return any(term in text for term in ("context window", "context_length_exceeded",
        "maximum context length", "too many tokens", "input exceeds", "input is too long"))


def get_client():
    global _client
    if _client is None:
        import httpx
        from openai import OpenAI

        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None,
            timeout=httpx.Timeout(connect=config.VLM_CONNECT_TIMEOUT, read=config.VLM_READ_TIMEOUT,
                                  write=120.0, pool=config.VLM_CONNECT_TIMEOUT),
            max_retries=0,
        )
    return _client


def patches(size) -> int:
    w, h = size
    return math.ceil(w / config.VLM_PATCH_PX) * math.ceil(h / config.VLM_PATCH_PX)


def fit_patch_limit(image: Image.Image):
    """(image, note). The image is returned UNCHANGED unless it exceeds the API's
    per-image patch limit; then it is scaled down just enough to be accepted."""
    if patches(image.size) <= config.VLM_MAX_PATCHES:
        return image, ""
    w, h = image.size
    scale = math.sqrt(config.VLM_MAX_PATCHES / patches(image.size))
    while patches((int(w * scale), int(h * scale))) > config.VLM_MAX_PATCHES:
        scale *= 0.995
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return small, ("%dx%d exceeds the API limit of %d patches -> sent at %dx%d"
                   % (w, h, config.VLM_MAX_PATCHES, small.size[0], small.size[1]))


def image_part(image: Image.Image, detail: str = "") -> dict:
    """An OpenAI `image_url` part with the image at its ORIGINAL size, lossless PNG
    (only an image over the API's patch limit is reduced, see fit_patch_limit)."""
    img, _ = fit_patch_limit(image.convert("RGB"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}
    if detail:
        url["detail"] = detail
    return {"type": "image_url", "image_url": url}


def _strip_detail(messages: List[dict]) -> bool:
    """Remove image_url "detail" keys; True if any were present."""
    found = False
    for message in messages:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url" and part["image_url"].pop("detail", None):
                    found = True
    return found


def _schema_prompt(schema: Type[BaseModel]) -> str:
    fields = {k: v for k, v in schema.model_json_schema().items() if k in ("properties", "required", "$defs")}
    return (
        "\n\nOUTPUT FORMAT\nReply with exactly ONE JSON object and nothing else "
        "(no prose, no markdown fences) that matches this JSON schema:\n"
        + json.dumps(fields, ensure_ascii=False)
    )


def _repair(text: str) -> str:
    """Common near-JSON slips: .5 instead of 0.5, trailing commas."""
    text = re.sub(r'(?<=[\[,:\s])(-?)\.(\d)', r'\g<1>0.\2', text)
    return re.sub(r",\s*([}\]])", r"\1", text)


def extract_json(text: str) -> dict:
    """First balanced top-level {...} in the text -- tolerates fences, a preamble
    and small slips. Never returns an object nested inside a broken outer one."""
    start = text.find("{")
    while start != -1:
        depth, in_str, escape, end = 0, False, False, None
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            break
        candidate = text[start:end]
        for attempt in (candidate, _repair(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        start = text.find("{", end)   # skip past the whole broken span
    raise LLMError("No valid JSON object in the model reply:\n%s" % text[:1500])


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


def _complete(body: dict, stream: bool) -> str:
    client = get_client()
    if not stream:
        response = client.chat.completions.create(**body)
        return response.choices[0].message.content or ""
    parts: List[str] = []
    for chunk in client.chat.completions.create(stream=True, **body):
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


def chat_json(
    model: str,
    messages: List[dict],
    schema: Type[BaseModel],
    json_mode: bool = False,
    temperature: Optional[float] = None,
    stream: bool = False,
    retries: int = 1,
) -> Tuple[BaseModel, str]:
    """Returns (validated object, raw reply text)."""
    messages = [dict(m) for m in messages]
    messages[0]["content"] = messages[0]["content"] + _schema_prompt(schema)

    body = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    attempt = 0
    while True:
        try:
            text = _complete(body, stream)
            break
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if is_context_window_error(exc):
                raise ContextWindowError("%s input exceeded the context window: %s" % (model, exc)) from exc
            if status == 400 and ("response_format" in body or "temperature" in body
                                  or _strip_detail(body["messages"])):
                body.pop("response_format", None)   # model rejects a parameter:
                body.pop("temperature", None)       # fall back to the bare request
                _strip_detail(body["messages"])     # (image "detail" included)
                continue
            if _is_transient(exc) and attempt < retries:
                attempt += 1
                time.sleep(3 * attempt)
                continue
            raise LLMError("%s call failed: %s" % (model, exc)) from exc

    try:
        return schema.model_validate(extract_json(text)), text
    except ValidationError as exc:
        raise LLMError("%s reply did not match the schema: %s\n%s" % (model, exc, text[:1500])) from exc
