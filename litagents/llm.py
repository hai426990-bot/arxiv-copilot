"""可选的 OpenAI 兼容 LLM 客户端。

未配置 API Key 时所有 Agent 自动使用确定性抽取逻辑，不影响主流程。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        api_key = os.environ.get("LITAGENTS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.environ.get("LITAGENTS_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_MODEL,
            base_url=(
                os.environ.get("LITAGENTS_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or DEFAULT_BASE_URL
            ).rstrip("/"),
            timeout=float(os.environ.get("LITAGENTS_TIMEOUT", "60")),
        )

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - 依赖已声明
            raise LLMError("未安装 requests，无法调用 LLM") from exc

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout,
            )
        except Exception as exc:
            raise LLMError(f"LLM 请求失败：{exc}") from exc

        if response.status_code != 200:
            raise LLMError(f"LLM HTTP {response.status_code}")
        try:
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM 返回格式异常") from exc

    def chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        text = self.chat(system, user)
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None
