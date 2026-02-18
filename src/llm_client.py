# src/llm_client.py
import os
import json
import time
from typing import Optional, Dict, Any, List

class LLMClient:
    """
    Minimal, provider-agnostic wrapper.
    Supports:
      - OpenAI (set OPENAI_API_KEY)
      - Ollama local (set OLLAMA_HOST and model like 'llama3.1')
    If neither is configured, calls will raise RuntimeError (the caller should fall back).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "")).lower()
        self.model = model or os.getenv("LLM_MODEL", "")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.timeout = 120.0
        self.max_retries = max_retries

        # autodetect if not explicitly set
        if not self.provider:
            if os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.getenv("OLLAMA_HOST"):
                self.provider = "ollama"

        # default models
        if not self.model:
            if self.provider == "openai":
                self.model = os.getenv("OPENAI_MODEL", "gpt-5.1")
            elif self.provider == "ollama":
                self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system"|"user"|"assistant", "content":"..."}]
        returns raw text content. Raises on failure.
        """
        if self.provider == "openai":
            return self._chat_openai(messages)
        if self.provider == "ollama":
            return self._chat_ollama(messages)
        raise RuntimeError("No LLM provider configured. Set OPENAI_API_KEY or OLLAMA_HOST.")

    # ---------- providers ----------

    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        import requests  # use requests to avoid hard dep on SDK
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }

        err = None
        for _ in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                err = e
                time.sleep(0.8)
        raise RuntimeError(f"OpenAI call failed: {err}")

    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        url = f"{host}/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }

        err = None
        for _ in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                # ollama’s openai-compatible chat endpoint returns same-ish structure
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                err = e
                time.sleep(0.8)
        raise RuntimeError(f"Ollama call failed: {err}")
