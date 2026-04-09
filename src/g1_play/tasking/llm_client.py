from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


class PlannerClientError(RuntimeError):
    pass


CompletionFn = Callable[[list[dict[str, str]], str, float, float, int], str]


def make_http_completion(base_url: str) -> CompletionFn:
    def complete(
        messages: list[dict[str, str]],
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
    ) -> str:
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            body["model"] = model
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            base_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PlannerClientError(f"LLM server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PlannerClientError(f"Could not reach LLM server at {base_url}. ({exc})") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerClientError(f"Unexpected LLM response shape: {payload}") from exc
        if not isinstance(content, str) or not content.strip():
            raise PlannerClientError("LLM response content was empty.")
        return content

    return complete


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    if start == -1:
        raise PlannerClientError("LLM response did not contain a JSON object.")
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx, char in enumerate(candidate[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise PlannerClientError("LLM response contained malformed JSON.")
    try:
        return json.loads(candidate[start:end])
    except json.JSONDecodeError as exc:
        raise PlannerClientError(f"LLM response contained invalid JSON: {exc}") from exc


def call_json_with_retry(
    completion: CompletionFn,
    messages: list[dict[str, str]],
    model: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    validator,
) -> dict[str, Any]:
    current_messages = list(messages)
    last_error: Exception | None = None
    last_content = ""
    for attempt in range(2):
        raw_content = completion(current_messages, model, timeout, temperature, max_tokens)
        last_content = raw_content
        try:
            parsed = extract_json_object(raw_content)
            return validator(parsed)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 1:
                break
            current_messages = list(messages) + [
                {"role": "assistant", "content": raw_content},
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid. "
                        f"Problem: {exc}. Reply again with exactly one valid JSON object and nothing else."
                    ),
                },
            ]
    raise PlannerClientError(
        f"LLM returned invalid JSON after 2 attempts. Last response: {last_content}"
    ) from last_error
