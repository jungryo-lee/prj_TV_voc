"""Shared Databricks LLM client helpers for prj_TV_voc."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from mlflow.deployments import get_deploy_client

from common.config_loader import load_config


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Extract a JSON object from a loose LLM response."""
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]

    return json.loads(raw)


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

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        llm_cfg = self.config.get("llm", {})

        self.provider = llm_cfg.get("provider", "databricks")
        self.endpoint = llm_cfg.get("endpoint", "databricks-claude-sonnet-4-6")
        self.max_retries = int(llm_cfg.get("max_retries", 3))
        self.backoff_base = float(llm_cfg.get("backoff_base", 1.8))
        self.timeout_seconds = int(llm_cfg.get("timeout_seconds", 60))
        self.inference_config = llm_cfg.get("inference", {})

        if self.provider != "databricks":
            raise ValueError(f"Unsupported llm.provider: {self.provider}")

        self.client = get_deploy_client("databricks")

    def build_inference_config(
        self,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Build inference config with optional overrides."""
        base = {
            "max_tokens": self.inference_config.get("max_tokens", 2200),
            "temperature": self.inference_config.get("temperature", 0.0),
        }

        if max_tokens is not None:
            base["max_tokens"] = max_tokens
        if temperature is not None:
            base["temperature"] = temperature
        if top_p is not None:
            base["top_p"] = top_p
        elif self.inference_config.get("top_p") is not None:
            base["top_p"] = self.inference_config.get("top_p")

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
        text = self.converse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return _parse_json_loose(text)


def get_llm_client(config: dict[str, Any] | None = None) -> DatabricksLLMClient:
    """Factory helper for downstream modules."""
    return DatabricksLLMClient(config=config)
