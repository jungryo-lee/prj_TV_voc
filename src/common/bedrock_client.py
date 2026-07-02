"""Shared AWS Bedrock client helpers for prj_TV_voc."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import boto3
from dotenv import load_dotenv

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


def _load_env_from_config(config: dict[str, Any]) -> None:
    """Load environment variables from the configured .env path if present."""
    env_path = config.get("path", {}).get("env")
    if env_path:
        load_dotenv(env_path)


class BedrockClient:
    """Thin wrapper around boto3 Bedrock runtime client."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        _load_env_from_config(self.config)

        bedrock_cfg = self.config.get("bedrock", {})
        self.region_name = os.getenv(
            bedrock_cfg.get("region_env", "AWS_REGION"),
            "ap-northeast-2",
        )
        self.aws_access_key_id = os.getenv(
            bedrock_cfg.get("access_key_env", "AWS_ACCESS_KEY_ID")
        )
        self.aws_secret_access_key = os.getenv(
            bedrock_cfg.get("secret_key_env", "AWS_SECRET_ACCESS_KEY")
        )

        self.model_id = bedrock_cfg.get("model_id")
        self.max_retries = int(bedrock_cfg.get("max_retries", 3))
        self.backoff_base = float(bedrock_cfg.get("backoff_base", 1.8))
        self.timeout_seconds = int(bedrock_cfg.get("timeout_seconds", 60))
        self.inference_config = bedrock_cfg.get("inference", {})

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

    def build_inference_config(
        self,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Build inference config with optional overrides."""
        base = {
            "maxTokens": self.inference_config.get("max_tokens", 2200),
            "temperature": self.inference_config.get("temperature", 0.0),
            "topP": self.inference_config.get("top_p", 0.9),
        }

        if max_tokens is not None:
            base["maxTokens"] = max_tokens
        if temperature is not None:
            base["temperature"] = temperature
        if top_p is not None:
            base["topP"] = top_p

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
        """Run a Bedrock converse call and return the raw text output."""
        inference_config = self.build_inference_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    inferenceConfig=inference_config,
                )
                return response["output"]["message"]["content"][0]["text"]
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt)

        raise RuntimeError(f"Bedrock converse failed: {last_error!r}")

    def converse_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Run a Bedrock call and parse the response as JSON."""
        text = self.converse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return _parse_json_loose(text)


def get_bedrock_client(config: dict[str, Any] | None = None) -> BedrockClient:
    """Factory helper for downstream modules."""
    return BedrockClient(config=config)
