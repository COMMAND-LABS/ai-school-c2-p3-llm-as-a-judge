"""Kalygo completion API client and stream parser."""

import json
from typing import Any
from urllib.parse import quote

import requests


def _extract_text_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("completion", "output", "text", "delta", "token", "answer", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if "message" in payload:
            return _extract_text_from_payload(payload["message"])
        if "choices" in payload and isinstance(payload["choices"], list):
            chunks = [_extract_text_from_payload(choice) for choice in payload["choices"]]
            return "".join(chunk for chunk in chunks if chunk)

    if isinstance(payload, list):
        chunks = [_extract_text_from_payload(item) for item in payload]
        return "".join(chunk for chunk in chunks if chunk)

    return ""


def _iter_json_objects(payload_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(payload_text)
    objects: list[dict[str, Any]] = []

    while idx < length:
        while idx < length and payload_text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            parsed, end_idx = decoder.raw_decode(payload_text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict):
            objects.append(parsed)
        idx = end_idx

    return objects


def call_kalygo_completion(
    *,
    api_url: str,
    api_key: str | None,
    agent_id: str,
    session_id: str,
    prompt: str,
    kalygo_api_timeout_seconds: int = 120,
    kalygo_api_retries: int = 2,
) -> str:
    url = f"{api_url.rstrip('/')}/api/agents/{quote(agent_id, safe='')}/completion"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    max_attempts = max(1, kalygo_api_retries + 1)
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"sessionId": session_id, "prompt": prompt},
                stream=True,
                timeout=kalygo_api_timeout_seconds,
            )

            if not response.ok:
                details = response.text
                raise RuntimeError(f"Kalygo API error {response.status_code}: {details}")

            stream_chunks: list[str] = []
            final_answer: str | None = None
            current_event = "message"

            def _handle_payload(payload_text: str, event_hint: str) -> str | None:
                nonlocal final_answer
                if payload_text == "[DONE]":
                    return "done"

                parsed_objects = _iter_json_objects(payload_text)
                for parsed in parsed_objects:
                    payload_event = parsed.get("event")
                    event_name = payload_event if isinstance(payload_event, str) and payload_event else event_hint

                    if event_name == "error":
                        error_data = parsed.get("data")
                        if isinstance(error_data, dict):
                            error_message = error_data.get("message") or error_data.get("error")
                        else:
                            error_message = parsed.get("message") or parsed.get("error") or error_data
                        raise RuntimeError(f"Kalygo stream error: {error_message or 'unknown error'}")

                    if event_name == "on_chain_end":
                        chain_data = parsed.get("data")
                        if isinstance(chain_data, str):
                            final_answer = chain_data.strip()
                        elif chain_data is not None:
                            final_answer = _extract_text_from_payload(chain_data).strip()
                        continue

                    if event_name == "on_chat_model_stream":
                        stream_piece = parsed.get("data")
                        if isinstance(stream_piece, str) and stream_piece:
                            stream_chunks.append(stream_piece)
                        continue

                if parsed_objects:
                    return None

                if event_hint == "error":
                    raise RuntimeError(f"Kalygo stream error: {payload_text}")
                return None

            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                stripped = line.strip()
                if not stripped:
                    current_event = "message"
                    continue
                if stripped.startswith("event:"):
                    current_event = stripped[6:].strip() or "message"
                    continue
                if stripped.startswith(("id:", "retry:")):
                    continue

                payload_text = stripped[5:].strip() if stripped.startswith("data:") else stripped
                outcome = _handle_payload(payload_text, current_event)
                if outcome == "done":
                    break

            if final_answer is not None:
                return final_answer
            return "".join(stream_chunks).strip()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                continue
            break

    raise RuntimeError(
        "Kalygo completion request failed after "
        f"{max_attempts} attempt(s) due to timeout/connection errors at {url}: {last_exc}"
    )


def fetch_kalygo_agent_config(
    *,
    api_url: str,
    api_key: str | None,
    agent_id: str,
    kalygo_api_timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Fetch agent metadata/config from Kalygo API."""
    url = f"{api_url.rstrip('/')}/api/agents/{quote(agent_id, safe='')}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(
        url,
        headers=headers,
        timeout=kalygo_api_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return {"raw": payload}
    return payload
