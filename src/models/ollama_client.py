"""
src/models/ollama_client.py
---------------------------
Clean, JSON-first Ollama client.

WHY JSON-FIRST:
  The assignment requires structured analysis and full logging. Free-form
  text responses make transcript parsing fragile. By prompting with
  explicit JSON instructions and parsing with fallback extraction, we get
  reliable structured outputs from any Ollama model.

FALLBACK PARSING CHAIN:
  1. Try json.loads() on full response
  2. Try extracting ```json ... ``` code block
  3. Try extracting first { ... } block
  4. Return empty dict (caller normalizes via validate_*_output)
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 120,
    ):
        self.base_url  = base_url.rstrip("/")
        self.model     = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    # ── Core generate ──────────────────────────────────────────────────
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send prompt, return raw text response."""
        payload = json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Run: ollama serve\n{e}"
            )

    # ── JSON generate ──────────────────────────────────────────────────
    def generate_json(self, prompt: str, temperature: Optional[float] = None) -> dict:
        """
        Generate and parse JSON response.
        Returns parsed dict, or empty dict on any parse failure.
        Caller is responsible for validation via validate_*_output().
        """
        raw = self.generate(prompt, temperature)
        return self._parse_json(raw)

    # ── Availability check ─────────────────────────────────────────────
    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=5
            ):
                return True
        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Check if the specific model is pulled."""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=5
            ) as resp:
                data = json.loads(resp.read())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False

    # ── JSON parsing with fallback chain ──────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Robust JSON extraction with 3-level fallback.
        Never raises — always returns dict (possibly empty).
        """
        if not text:
            return {}

        # 1. Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # 2. Extract ```json ... ``` block
        block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if block:
            try:
                result = json.loads(block.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # 3. Extract first { ... } block (greedy)
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                result = json.loads(brace.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # 4. Nothing worked — return empty dict, caller will normalize
        return {}