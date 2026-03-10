import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_seconds: int = 45,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.2,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 700,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        endpoint = f"{self.base_url}/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            req = request.Request(endpoint, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    parsed = json.loads(body)
                    choices = parsed.get("choices", [])
                    content = ""
                    if choices and isinstance(choices, list):
                        content = choices[0].get("message", {}).get("content", "")

                    return LLMResponse(
                        content=content,
                        model=parsed.get("model", self.model),
                        usage=parsed.get("usage", {}),
                        raw=parsed,
                    )
            except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep_seconds = self.retry_backoff_seconds * (attempt + 1)
                    time.sleep(sleep_seconds)
                else:
                    break

        raise RuntimeError(f"DeepSeek API 调用失败: {last_error}")
