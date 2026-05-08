"""
ai/bearer_pool.py — Bearer Token Pool with automatic rotation
Manages multiple Qwen accounts for load distribution & rate limit avoidance
"""
import asyncio
import logging
import time
from typing import Optional
from collections import deque

import config
from ai.qwen_client import QwenClient

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from Qwen response.
    Handles:
    - Markdown fences (```json ... ```)
    - Extra text before/after JSON
    - Multiple JSON objects (takes first complete one)
    - Python dict syntax with single quotes  ← NEW
    - Trailing commas                        ← NEW
    - None/True/False Python literals        ← NEW
    """
    import json
    import re
    import ast

    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw = "\n".join(inner).strip()

    # ── Attempt 1: direct JSON parse (happy path) ─────────────────────────
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ── Find the outermost { ... } block ─────────────────────────────────
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")

    # Track brace depth to find the matching closing brace
    depth = 0
    in_str = False
    esc = False
    candidate = raw  # fallback
    for i, ch in enumerate(raw[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:i + 1]
                break

    # ── Attempt 2: JSON parse the extracted block ─────────────────────────
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # ── Attempt 3: fix Python dict → JSON ────────────────────────────────
    # Qwen sometimes returns Python dict syntax: {'key': 'value', 'n': None}
    # ast.literal_eval handles single quotes, None, True, False
    try:
        py_obj = ast.literal_eval(candidate)
        if isinstance(py_obj, dict):
            # Round-trip through json to normalize types
            return json.loads(json.dumps(py_obj))
    except Exception:
        pass

    # ── Attempt 4: aggressive cleanup then re-parse ───────────────────────
    try:
        fixed = candidate
        # single quotes → double quotes (careful not to break already-double-quoted)
        fixed = re.sub(r"(?<![\\])'", '"', fixed)
        # Python None/True/False → JSON null/true/false
        fixed = re.sub(r'\bNone\b', 'null', fixed)
        fixed = re.sub(r'\bTrue\b', 'true', fixed)
        fixed = re.sub(r'\bFalse\b', 'false', fixed)
        # trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        return json.loads(fixed)
    except Exception:
        pass

    raise ValueError(
        f"Could not parse JSON from response after all attempts.\n"
        f"Candidate block: {candidate[:400]}"
    )


class BearerPool:
    """
    Manages a pool of Qwen bearer tokens.
    Rotates tokens automatically, tracks failures, cools down bad tokens.
    """

    def __init__(self, bearers: Optional[list[str]] = None, model: str = None):
        self.model = model or config.QWEN_MODEL
        raw_bearers = bearers or config.QWEN_BEARERS

        if not raw_bearers:
            raise ValueError(
                "No Qwen bearer tokens configured. "
                "Set QWEN_BEARERS in .env"
            )

        self._tokens = deque(raw_bearers)
        self._clients: dict[str, QwenClient] = {
            t: QwenClient(t, self.model) for t in raw_bearers
        }
        self._failures: dict[str, int] = {t: 0 for t in raw_bearers}
        self._cooldowns: dict[str, float] = {t: 0.0 for t in raw_bearers}
        self._lock = asyncio.Lock()
        self._current_idx = 0

        logger.info(f"[BearerPool] Initialized with {len(raw_bearers)} tokens")

    def _is_available(self, token: str) -> bool:
        cooldown_until = self._cooldowns.get(token, 0)
        if time.time() < cooldown_until:
            return False
        if self._failures.get(token, 0) >= 5:
            # Reset after 10 minutes
            if time.time() > cooldown_until + 600:
                self._failures[token] = 0
                self._cooldowns[token] = 0
            else:
                return False
        return True

    def _get_next_available(self) -> Optional[str]:
        tokens_list = list(self._tokens)
        n = len(tokens_list)
        for i in range(n):
            idx = (self._current_idx + i) % n
            token = tokens_list[idx]
            if self._is_available(token):
                self._current_idx = (idx + 1) % n
                return token
        return None

    def _mark_failure(self, token: str):
        self._failures[token] = self._failures.get(token, 0) + 1
        failures = self._failures[token]
        if failures >= 3:
            cooldown_minutes = min(failures * 2, 30)
            self._cooldowns[token] = time.time() + cooldown_minutes * 60
            logger.warning(
                f"[BearerPool] Token ...{token[-8:]} in cooldown "
                f"for {cooldown_minutes}m (failures: {failures})"
            )

    def _mark_success(self, token: str):
        self._failures[token] = max(0, self._failures.get(token, 0) - 1)
        self._cooldowns[token] = 0

    async def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
        retries: int = None,
        web_search: bool = False,
    ) -> str:
        """Ask Qwen using pool rotation with auto-retry on failure."""
        max_retries = retries or config.MAX_RETRIES

        for attempt in range(max_retries):
            async with self._lock:
                token = self._get_next_available()

            if not token:
                wait = 30 * (attempt + 1)
                logger.warning(
                    f"[BearerPool] All tokens unavailable, "
                    f"waiting {wait}s (attempt {attempt+1})"
                )
                await asyncio.sleep(wait)
                continue

            client = self._clients[token]
            try:
                result = await client.ask(
                    prompt,
                    system_prompt=system_prompt,
                    history=history,
                    web_search=web_search,
                )
                async with self._lock:
                    self._mark_success(token)
                return result

            except Exception as e:
                async with self._lock:
                    self._mark_failure(token)
                logger.warning(
                    f"[BearerPool] Token ...{token[-8:]} failed "
                    f"(attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"[BearerPool] All {max_retries} attempts failed")

    async def ask_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
    ) -> dict:
        """Ask Qwen and return parsed JSON — robust against extra text/multiple objects."""
        json_system = (system_prompt or "") + (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no explanation, no backticks. Pure JSON only."
        )
        raw = await self.ask(prompt, system_prompt=json_system, history=history)
        return _extract_json(raw)

    @property
    def status(self) -> dict:
        return {
            "total_tokens": len(self._clients),
            "available": sum(1 for t in self._clients if self._is_available(t)),
            "in_cooldown": sum(
                1 for t, cd in self._cooldowns.items()
                if time.time() < cd
            ),
        }


# Global singleton
_pool: Optional[BearerPool] = None


def get_pool() -> BearerPool:
    global _pool
    if _pool is None:
        _pool = BearerPool()
    return _pool
