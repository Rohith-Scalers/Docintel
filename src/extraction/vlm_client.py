"""Async DeepSeek-OCR-2 (OpenAI-compatible) VLM client with retry and prompt management."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.config import VLMConfig
from src.models import ExtractionSchema, RawRegion, RegionType

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE: dict[str, Any] = {
    "raw_text": "",
    "overview": "",
    "confidence": 0.0,
}

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class VLMClient:
    """Async client for the DeepSeek-OCR-2 OpenAI-compatible endpoint.

    Loads prompt templates from the YAML file specified in
    VLMConfig.prompt_config_path. Each region type maps to a system prompt and
    an overview instruction. A shared extraction_prompt_suffix is appended to
    every non-overview extraction call.

    The underlying httpx.AsyncClient is reused across requests. Call aclose()
    when the client is no longer needed, or use it as an async context manager.
    """

    def __init__(
        self,
        config: VLMConfig,
        prompts: dict[str, Any],
    ) -> None:
        self._config = config
        self._prompts = prompts
        self._client = httpx.AsyncClient(timeout=config.timeout_s)

    @classmethod
    def from_config(cls, config: VLMConfig) -> "VLMClient":
        """Construct a VLMClient by loading prompt templates from the config path.

        Returns:
            VLMClient: configured client instance with prompts loaded from YAML.
        """
        prompt_path = Path(config.prompt_config_path)
        with prompt_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(config=config, prompts=raw)

    async def extract(
        self,
        region: RawRegion,
        schema: ExtractionSchema | None,
        max_tokens: int,
    ) -> dict:
        """Extract structured content from a region image using the VLM.

        Builds a system prompt from the region-type template plus the shared
        extraction_prompt_suffix. If an ExtractionSchema is provided, appends
        field-level instructions. Sends a vision request with the cropped image
        encoded as base64 data URI and parses the JSON response.

        Returns:
            dict: parsed JSON extraction result, or fallback dict on parse
            failure ({"raw_text": "", "overview": "", "confidence": 0.0}).
        """
        system_prompt = self.build_extraction_prompt(region.region_type, schema)
        return await self.extract_with_system_prompt(region, system_prompt, max_tokens)

    async def extract_with_system_prompt(
        self,
        region: RawRegion,
        system_prompt: str,
        max_tokens: int,
    ) -> dict:
        """Extract content using a fully-formed system prompt (bypass template building).

        Used by the Validator during correction re-prompts when the system
        prompt has already been augmented with correction hints.

        Returns:
            dict: parsed JSON extraction result, or fallback dict on parse failure.
        """
        image_b64 = _encode_image(region.cropped_image_path)
        messages = _build_messages(system_prompt, image_b64)
        payload = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        raw = await self._post_with_retry(payload)
        return self._parse_json_response(raw)

    async def extract_overview(self, region: RawRegion) -> str:
        """Generate a 1-2 sentence plain-English description of a region.

        Uses only the overview_instruction for the given region type, keeping
        token usage minimal. Falls back to an empty string on any failure.

        Returns:
            str: overview description text.
        """
        region_key = region.region_type.value
        region_prompts = self._prompts.get("prompts", {}).get(region_key, {})
        instruction = region_prompts.get(
            "overview_instruction",
            "In 1-2 sentences describe what this region contains.",
        )
        image_b64 = _encode_image(region.cropped_image_path)
        messages = _build_messages(instruction, image_b64)
        payload = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.overview_max_tokens,
        }
        raw = await self._post_with_retry(payload)
        content = _extract_text_content(raw)
        return content.strip()

    def build_extraction_prompt(
        self,
        region_type: RegionType,
        schema: ExtractionSchema | None,
    ) -> str:
        """Build the full system prompt for an extraction call.

        Assembles: region-type system text + overview instruction + shared
        extraction suffix + any ExtractionSchema field instructions.

        Returns:
            str: assembled system prompt string.
        """
        region_key = region_type.value
        region_prompts = self._prompts.get("prompts", {}).get(region_key, {})
        parts: list[str] = [
            region_prompts.get("system", "Extract all content from this region.").strip()
        ]

        overview_instruction = region_prompts.get("overview_instruction", "").strip()
        if overview_instruction:
            parts.append(overview_instruction)

        suffix = self._prompts.get("extraction_prompt_suffix", "").strip()
        if suffix:
            parts.append(suffix)

        if schema and schema.fields:
            field_lines = "\n".join(
                f'  "{k}": "{v}"' for k, v in schema.fields.items()
            )
            parts.append(
                f"Also extract the following schema fields as JSON keys:\n{field_lines}"
            )

        if schema and schema.extract_entities and schema.entity_types:
            types_str = "|".join(schema.entity_types)
            parts.append(
                f"Restrict entity types to: {types_str}. "
                "Return them in the 'entities' key as described above."
            )

        return "\n\n".join(parts)

    async def _post_with_retry(self, payload: dict) -> dict:
        """POST to the VLM endpoint with exponential backoff on 429/5xx.

        Retries up to VLMConfig.max_retries times. Wait between attempts is
        2^attempt seconds. Logs a warning on each retried status code and an
        error when all attempts are exhausted.

        Returns:
            dict: raw JSON response from the endpoint, or empty dict when all
            retry attempts are exhausted.
        """
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.post(
                    self._config.endpoint,
                    json=payload,
                    headers=headers,
                )
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    wait_s = 2 ** attempt
                    logger.warning(
                        "VLM returned %d on attempt %d/%d, retrying in %ds",
                        response.status_code,
                        attempt + 1,
                        self._config.max_retries,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.TransportError as exc:
                last_exc = exc
                wait_s = 2 ** attempt
                logger.warning(
                    "VLM transport error on attempt %d/%d: %s, retrying in %ds",
                    attempt + 1,
                    self._config.max_retries,
                    exc,
                    wait_s,
                )
                await asyncio.sleep(wait_s)

        logger.error(
            "VLM request failed after %d attempts. Last error: %s",
            self._config.max_retries,
            last_exc,
        )
        return {}

    def _parse_json_response(self, response: dict) -> dict:
        """Parse JSON from the assistant message content field.

        Strips markdown code fences (```json ... ```) if present before
        JSON parsing. Returns the fallback dict on any parse failure.

        Returns:
            dict: parsed result dict, or fallback on failure.
        """
        content = _extract_text_content(response)
        if not content:
            return dict(_FALLBACK_RESPONSE)

        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            stripped = "\n".join(lines[1:end]).strip()

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            logger.warning(
                "VLM response parsed as %s (expected dict); using fallback.",
                type(parsed).__name__,
            )
            return dict(_FALLBACK_RESPONSE)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed for VLM response: %s; using fallback.", exc)
            return dict(_FALLBACK_RESPONSE)

    async def aclose(self) -> None:
        """Close the underlying httpx async client."""
        await self._client.aclose()

    async def __aenter__(self) -> "VLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def _encode_image(path: str) -> str:
    """Read and base64-encode a cropped image file.

    Returns:
        str: base64-encoded image bytes as an ASCII string.
    """
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _build_messages(system_prompt: str, image_b64: str) -> list[dict]:
    """Build an OpenAI-compatible messages list with a system turn and a vision user turn.

    Returns:
        list[dict]: messages array ready for the chat completions API.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                }
            ],
        },
    ]


def _extract_text_content(response: dict) -> str:
    """Extract the assistant's text content from a chat completions response.

    Returns:
        str: the text content string, or empty string if the response is
        malformed or missing the expected structure.
    """
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
