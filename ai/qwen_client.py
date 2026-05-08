"""
ai/qwen_client.py — Qwen Reverse Engineering Client
Adapted from testt.py — supports streaming, multi-turn, auto-retry
"""
import json
import uuid
import time
import asyncio
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QwenClient:
    """
    Async-compatible Qwen reverse engineering client.
    Uses requests under the hood with asyncio.to_thread for non-blocking calls.
    """

    BASE_URL = "https://chat.qwen.ai"

    def __init__(self, bearer_token: str, model: str = "qwen3-max"):
        self.bearer_token = bearer_token
        self.model = model
        self.session = requests.Session()
        self._base_headers = {
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/json",
            "source": "web",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
            "Origin": "https://chat.qwen.ai",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Version": "0.2.7",
            "bx-v": "2.5.36",
            "Authorization": f"Bearer {self.bearer_token}",
        }

    def _make_uuid(self) -> str:
        return str(uuid.uuid4())

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {**self._base_headers, "X-Request-Id": self._make_uuid()}
        if extra:
            h.update(extra)
        return h

    # ─── Sync helpers (run in thread) ─────────────────────────────────────────

    def _create_chat_sync(self) -> str:
        url = f"{self.BASE_URL}/api/v2/chats/new"
        payload = {
            "title": "TradingLab Agent Chat",
            "models": [self.model],
            "chat_mode": "normal",
            "chat_type": "t2t",
            "timestamp": int(time.time() * 1000),
            "project_id": "",
        }
        resp = self.session.post(url, json=payload, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        chat_id = resp.json()["data"]["id"]
        logger.debug(f"[Qwen] Chat created: {chat_id}")
        return chat_id

    def _send_message_sync(
        self,
        chat_id: str,
        message: str,
        history: Optional[list] = None,
        web_search: bool = False,
    ) -> str:
        url = f"{self.BASE_URL}/api/v2/chat/completions?chat_id={chat_id}"
        fid = self._make_uuid()
        child_id = self._make_uuid()
        ts = int(time.time())

        messages = []

        # inject history if any
        if history:
            for item in history:
                messages.append({
                    "fid": self._make_uuid(),
                    "parentId": None,
                    "childrenIds": [],
                    "role": item["role"],
                    "content": item["content"],
                    "user_action": "chat",
                    "files": [],
                    "timestamp": ts,
                    "models": [self.model],
                    "chat_type": "t2t",
                    "feature_config": {},
                    "extra": {"meta": {"subChatType": "t2t"}},
                    "sub_chat_type": "t2t",
                    "parent_id": None,
                })

        messages.append({
            "fid": fid,
            "parentId": None,
            "childrenIds": [child_id],
            "role": "user",
            "content": message,
            "user_action": "chat",
            "files": [],
            "timestamp": ts,
            "models": [self.model],
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": True,
                "output_schema": "phase",
                "research_mode": "normal",
                "auto_thinking": True,
                "thinking_mode": "Auto",
                "thinking_format": "summary",
                "auto_search": web_search,
            },
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t",
            "parent_id": None,
        })

        payload = {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": self.model,
            "parent_id": None,
            "messages": messages,
            "timestamp": ts + 1,
        }

        headers = self._headers({
            "Referer": f"https://chat.qwen.ai/c/{chat_id}",
            "x-accel-buffering": "no",
        })

        full_reply = ""
        think_buf  = ""   # Thinking phase — dikumpulkan tapi TIDAK masuk ke reply

        with self.session.post(
            url, json=payload, headers=headers, stream=True, timeout=180
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if not data.get("choices"):
                        continue
                    delta   = data["choices"][0].get("delta", {})
                    phase   = delta.get("phase", "answer")
                    content = delta.get("content", "")
                    status  = delta.get("status", "")

                    if phase == "thinking_summary":
                        # isi thinking Qwen — kumpulkan terpisah, JANGAN masuk reply
                        # dan JANGAN break di sini meskipun status finished
                        if content:
                            think_buf += content
                    else:
                        # Phase "answer" — ini yang kita mau
                        if content:
                            full_reply += content
                        # break HANYA saat answer finished, bukan thinking finished
                        if status == "finished":
                            break

                except (json.JSONDecodeError, KeyError):
                    pass

        if think_buf:
            logger.debug(f"[Qwen thinking] {think_buf[:200]}")
        # Edge case: Qwen hanya kirim thinking tanpa answer — fallback
        if not full_reply.strip() and think_buf:
            logger.warning("[Qwen] Answer kosong, fallback ke thinking content")
            full_reply = think_buf

        return full_reply

    def _delete_chat_sync(self, chat_id: str):
        url = f"{self.BASE_URL}/api/v2/chats/{chat_id}"
        try:
            self.session.delete(url, headers=self._headers(), timeout=30)
        except Exception:
            pass

    # ─── Async public API ──────────────────────────────────────────────────────

    async def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
        cleanup: bool = True,
        web_search: bool = False,
    ) -> str:
        """
        Send a prompt to Qwen and return the full response.
        Creates a new chat, sends message, optionally deletes chat.
        """
        loop = asyncio.get_event_loop()

        # Create chat
        chat_id = await loop.run_in_executor(None, self._create_chat_sync)

        # Build history with optional system prompt
        msg_history = []
        if system_prompt:
            msg_history.append({"role": "system", "content": system_prompt})
        if history:
            msg_history.extend(history)

        # Send message
        reply = await loop.run_in_executor(
            None,
            lambda: self._send_message_sync(chat_id, prompt, msg_history or None, web_search)
        )

        # Cleanup
        if cleanup:
            await loop.run_in_executor(None, lambda: self._delete_chat_sync(chat_id))

        logger.debug(f"[Qwen] Reply length: {len(reply)} chars")
        return reply

    async def ask_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
    ) -> dict:
        """Ask Qwen and return parsed JSON — robust extraction."""
        from ai.bearer_pool import _extract_json
        json_system = (system_prompt or "") + (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no explanation, no backticks. Pure JSON only."
        )
        raw = await self.ask(prompt, system_prompt=json_system, history=history)
        return _extract_json(raw)
