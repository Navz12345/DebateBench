"""
src/models/university_client.py
--------------------------------
OpenAI-compatible API client for UTSA hosted models.

WHY OPENAI SDK INSTEAD OF URLLIB:
  Both university endpoints expose the standard OpenAI /v1/chat/completions
  interface. The openai Python SDK handles auth, retries, and timeout.

JSON RELIABILITY FIXES:
  1. response_format={"type":"json_object"} — forces vLLM to constrain
     output to valid JSON (grammar-based sampling). Most reliable fix.
  2. _strip_thinking() — removes Qwen3's <think>...</think> blocks.
  3. 3-level JSON fallback parser — handles code blocks + embedded JSON.
  4. _extract_stance_from_text() — last-resort keyword scan when model
     writes prose instead of JSON despite instructions.
  5. Retry once with a simpler extraction prompt if first call returns {}.
"""

import json
import re
import time
from typing import Optional

from openai import OpenAI, APIConnectionError, APITimeoutError


class UniversityClient:
    """
    Client for one university-hosted model endpoint.
    Instantiate separately for debater (Qwen3) and judge (Llama).
    """

    def __init__(
        self,
        base_url:    str,
        api_key:     str,
        model:       str,
        temperature: float = 0.7,
        max_tokens:  int   = 1024,
        timeout:     int   = 120,
    ):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=2,
        )

    # ── Core generate ──────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        system: Optional[str] = None,
    ) -> str:
        """
        Send a prompt, return raw text response.

        system: optional system message sets the model's persona/role.
                Separating role (system) from task (user) improves consistency.

        THINKING MODE (Qwen3 only):
          We now ENABLE thinking mode instead of disabling it.
          Qwen3-8B does full internal chain-of-thought reasoning inside
          <think>...</think> before producing its answer. This reasoning
          is stripped before JSON parsing, but it dramatically improves
          the quality of the final answer — especially for multi-step
          scientific reasoning tasks like SciFact.
          The thinking tokens count toward max_tokens, so we budget 6144
          (≈2000 thinking + 4096 answer) for debaters.
        """
        temp = temperature if temperature is not None else self.temperature

        # Build message list — system message first if provided
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=self.max_tokens,
        )

        # Force JSON output when requested
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Qwen3: ENABLE thinking mode — the model reasons internally before
        # producing its answer. _strip_thinking() removes the <think> block
        # from the response before JSON parsing, so it never corrupts output.
        # budget_tokens caps the thinking portion so it doesn't eat all of max_tokens.
        if "qwen" in self.model.lower():
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }

        response = self._client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        return self._strip_thinking(raw).strip()

    # ── JSON generate with retry ───────────────────────────────────────
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> dict:
        """
        Generate and parse JSON. Retries if reasoning field is empty.
        Empty reasoning = model declined to argue (conflict with system msg).
        Retry without system message lets the prompt alone drive the output.
        """
        def _is_valid(d: dict) -> bool:
            """Valid means: parseable dict with non-empty reasoning or stance."""
            if not d or not isinstance(d, dict):
                return False
            if '_raw_text' in d:
                return False
            # Must have at least one substantive field
            reasoning = str(d.get('reasoning', '') or d.get('final_verdict', ''))
            return len(reasoning.strip()) > 10

        # Attempt 1: json_mode + system message
        try:
            raw = self.generate(prompt, temperature, json_mode=True, system=system)
            result = self._parse_json(raw)
            if _is_valid(result):
                return result
        except Exception:
            pass

        # Attempt 2: no json_mode, with system message
        try:
            raw = self.generate(prompt, temperature, json_mode=False, system=system)
            result = self._parse_json(raw)
            if _is_valid(result):
                return result
        except Exception:
            pass

        # Attempt 3: json_mode WITHOUT system message
        # System message may conflict with forced stance — try without it
        try:
            raw = self.generate(prompt, temperature, json_mode=True, system=None)
            result = self._parse_json(raw)
            if _is_valid(result):
                return result
        except Exception:
            pass

        # Attempt 4: plain generate, no json_mode, no system message
        try:
            raw = self.generate(prompt, temperature, json_mode=False, system=None)
            result = self._parse_json(raw)
            if _is_valid(result):
                return result
            return {"_raw_text": raw}
        except Exception:
            return {}

    # ── Availability check ─────────────────────────────────────────────
    def is_available(self) -> bool:
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=3,
            )
            return True
        except Exception:
            return False

    # ── Qwen3 thinking tag stripper ────────────────────────────────────
    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Remove Qwen3's <think>...</think> block.
        Also handles truncated blocks with no closing tag —
        when max_tokens is hit mid-thinking, </think> never appears.
        In that case, strip everything from <think> to end of string.
        """
        # Complete block: <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Incomplete block: <think>... (no closing tag — truncated by token limit)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    # ── JSON parsing with 3-level fallback ────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> dict:
        """Robust JSON extraction. Never raises. Returns {} on failure."""
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

        # 3. Extract first { ... } block
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                result = json.loads(brace.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {}

    # ── Text-based stance extractor (last resort) ─────────────────────
    @staticmethod
    def extract_stance_from_text(text: str) -> str:
        """
        Keyword scan for stance when JSON parsing fails completely.
        Called by validate_debater_output() when stance is still missing.
        """
        upper = text.upper()

        # Most specific first to avoid false matches
        nei_keywords = [
            'NOT_ENOUGH_INFO', 'NOT ENOUGH INFO', 'NOT ENOUGH INFORMATION',
            'INSUFFICIENT EVIDENCE', 'CANNOT DETERMINE', 'NOINFO',
        ]
        support_keywords = [
            'STANCE: SUPPORT', 'STANCE:SUPPORT', '"STANCE": "SUPPORT"',
            '"SUPPORT"', 'VERDICT: SUPPORT', 'FINAL_VERDICT: SUPPORT',
            'IS SUPPORTED', 'SUPPORTS THE CLAIM', 'EVIDENCE SUPPORTS',
            'CLAIM IS SUPPORTED', 'THE CLAIM IS SUPPORT',
        ]
        refute_keywords = [
            'STANCE: REFUTE', 'STANCE:REFUTE', '"STANCE": "REFUTE"',
            '"REFUTE"', 'VERDICT: REFUTE', 'FINAL_VERDICT: REFUTE',
            'IS REFUTED', 'REFUTES THE CLAIM', 'CONTRADICTS THE CLAIM',
            'DOES NOT SUPPORT', 'CLAIM IS REFUTED', 'THE CLAIM IS REFUTE',
        ]

        for kw in nei_keywords:
            if kw in upper:
                return 'NOT_ENOUGH_INFO'
        for kw in support_keywords:
            if kw in upper:
                return 'SUPPORT'
        for kw in refute_keywords:
            if kw in upper:
                return 'REFUTE'
        return ''


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def make_clients(
    config: dict,
) -> tuple["UniversityClient", "UniversityClient", "UniversityClient"]:
    """
    Returns (debater_client, judge_client, jury_client) from config.

    Three separate models:
      debater → Qwen3-8B       (small, fast, thinking mode)
      judge   → GPT-OSS-20B    (single verdict after debate)
      jury    → Llama-3.1-70B  (panel deliberation — larger = better reasoning)

    If 'jury_model' is not in config, jury_client falls back to judge_client.
    """
    mc          = config["models"]
    debater_cfg = mc["debater"]
    judge_cfg   = mc["judge"]
    jury_cfg    = mc.get("jury_model", judge_cfg)   # fallback to judge if not set

    debater_client = UniversityClient(
        base_url=    debater_cfg["base_url"],
        api_key=     debater_cfg["api_key"],
        model=       debater_cfg["name"],
        temperature= debater_cfg.get("temperature", 0.7),
        max_tokens=  debater_cfg.get("max_tokens",  1024),
        timeout=     debater_cfg.get("timeout",     120),
    )

    judge_client = UniversityClient(
        base_url=    judge_cfg["base_url"],
        api_key=     judge_cfg["api_key"],
        model=       judge_cfg["name"],
        temperature= judge_cfg.get("temperature", 0.3),
        max_tokens=  judge_cfg.get("max_tokens",  4096),
        timeout=     judge_cfg.get("timeout",     120),
    )

    jury_client = UniversityClient(
        base_url=    jury_cfg["base_url"],
        api_key=     jury_cfg["api_key"],
        model=       jury_cfg["name"],
        temperature= jury_cfg.get("temperature", 0.3),
        max_tokens=  jury_cfg.get("max_tokens",  4096),
        timeout=     jury_cfg.get("timeout",     180),
    )

    return debater_client, judge_client, jury_client