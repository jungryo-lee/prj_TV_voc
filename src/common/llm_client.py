"""Shared Databricks LLM client helpers for prj_TV_voc."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from common.config_loader import load_config


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Extract a JSON object from a loose LLM response."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM returned empty text; cannot parse JSON.")

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    raw = re.sub(r"^\s*json\s*", "", raw, flags=re.IGNORECASE).strip()

    candidates = [raw]

    obj_start = raw.find("{")
    obj_end = raw.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(raw[obj_start : obj_end + 1])

    arr_start = raw.find("[")
    arr_end = raw.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(raw[arr_start : arr_end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except Exception as exc:
            last_error = exc

    preview = raw[:500].replace("\n", "\\n")
    raise ValueError(
        f"Failed to parse JSON from LLM response. preview={preview!r}, error={last_error!r}"
    )


def _extract_text_from_response(response: Any) -> str:
    """Normalize common Databricks endpoint response shapes."""
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        if response.get("choices"):
            return response["choices"][0]["message"]["content"]
        if response.get("predictions"):
            prediction = response["predictions"][0]
            if isinstance(prediction, dict) and "content" in prediction:
                return prediction["content"]
            if isinstance(prediction, str):
                return prediction
        if isinstance(response.get("content"), str):
            return response["content"]

    raise ValueError(f"Unexpected Databricks LLM response schema: {response!r}")


class DatabricksLLMClient:
    """Thin wrapper around Databricks hosted LLM endpoints."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_key: str | None = None,
    ):
        self.config = config or load_config()
        llm_cfg = self.config.get("llm", {})

        self.provider = llm_cfg.get("provider", "databricks")
        self.max_retries = int(llm_cfg.get("max_retries", 3))
        self.backoff_base = float(llm_cfg.get("backoff_base", 1.8))
        self.timeout_seconds = int(llm_cfg.get("timeout_seconds", 60))

        if self.provider != "databricks":
            raise ValueError(f"Unsupported llm.provider: {self.provider}")

        selected_model_key = model_key or llm_cfg.get("default_model_key")
        if not selected_model_key:
            raise ValueError("llm.default_model_key or model_key is required.")

        models_cfg = llm_cfg.get("models", {})
        model_cfg = models_cfg.get(selected_model_key)
        if not model_cfg:
            raise KeyError(f"Unknown model_key: {selected_model_key}")

        self.model_key = selected_model_key
        self.endpoint = model_cfg["endpoint"]
        self.model_version = model_cfg.get("model_version", selected_model_key)
        self.inference_config = model_cfg.get("inference", {})

        try:
            from mlflow.deployments import get_deploy_client
        except ImportError as exc:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "mlflow is required for Databricks LLM calls. "
                "Install mlflow or run this client in Databricks Runtime."
            ) from exc

        self.client = get_deploy_client("databricks")

    def build_inference_config(
        self,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Build inference config with optional overrides."""
        base: dict[str, Any] = {
            "max_tokens": self.inference_config.get("max_tokens", 2200),
        }

        default_temperature = self.inference_config.get("temperature")
        default_top_p = self.inference_config.get("top_p")

        if temperature is not None:
            base["temperature"] = temperature
        elif default_temperature is not None:
            base["temperature"] = default_temperature

        if top_p is not None:
            if "temperature" in base:
                base.pop("temperature", None)
            base["top_p"] = top_p
        elif default_top_p is not None and "temperature" not in base:
            base["top_p"] = default_top_p

        if max_tokens is not None:
            base["max_tokens"] = max_tokens

        return base

    def converse(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Run a Databricks hosted LLM call and return the raw text output."""
        inference_config = self.build_inference_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **inference_config,
        }

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.predict(endpoint=self.endpoint, inputs=payload)
                return _extract_text_from_response(response)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt)

        raise RuntimeError(f"Databricks LLM call failed: {last_error!r}")

    def converse_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Run a Databricks LLM call and parse the response as JSON."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            text = self.converse(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            try:
                return _parse_json_loose(text)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt)

        raise RuntimeError(f"Databricks LLM JSON parse failed: {last_error!r}")


def get_llm_client(
    config: dict[str, Any] | None = None,
    model_key: str | None = None,
) -> DatabricksLLMClient:
    """Factory helper for downstream modules."""
    return DatabricksLLMClient(config=config, model_key=model_key)
