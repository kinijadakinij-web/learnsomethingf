"""
core/agent_base.py — Abstract base class for all Trading Lab agents
Every agent inherits from this to get: Qwen access, event bus, logging, lifecycle
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from ai.bearer_pool import BearerPool, get_pool
from core.event_bus import EventBus, Event, EventType, get_bus

logger = logging.getLogger(__name__)


class AgentStatus:
    IDLE       = "idle"
    THINKING   = "thinking"
    WORKING    = "working"
    WAITING    = "waiting"
    ERROR      = "error"
    STOPPED    = "stopped"


class BaseAgent(ABC):
    """
    Base class for all agents in the Trading Lab ecosystem.

    Each agent has:
    - A unique agent_id and role name
    - Access to the Qwen bearer pool for AI calls
    - Access to the event bus for communication
    - Lifecycle management (start/stop)
    - Status tracking
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        bearer_pool: Optional[BearerPool] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.pool = bearer_pool or get_pool()
        self.bus = event_bus or get_bus()
        self.status = AgentStatus.IDLE
        self.created_at = time.time()
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._running = False
        self.logger = logging.getLogger(f"agent.{role}.{agent_id[:8]}")

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self):
        """Start the agent — register event subscriptions and announce."""
        self._running = True
        self._register_subscriptions()
        await self.bus.emit(
            EventType.AGENT_STARTED,
            source_agent=self.agent_id,
            payload={"role": self.role, "agent_id": self.agent_id},
        )
        self.logger.info(f"[{self.role}] Agent started: {self.agent_id[:8]}")
        await self.on_start()

    async def stop(self):
        """Stop the agent gracefully."""
        self._running = False
        self.status = AgentStatus.STOPPED
        await self.bus.emit(
            EventType.AGENT_STOPPED,
            source_agent=self.agent_id,
            payload={"role": self.role, "agent_id": self.agent_id},
        )
        self.logger.info(f"[{self.role}] Agent stopped")

    async def on_start(self):
        """Override for post-start initialization logic."""
        pass

    @abstractmethod
    def _register_subscriptions(self):
        """Subscribe to relevant events on the bus."""
        pass

    # ─── AI Calls ───────────────────────────────────────────────────────────

    async def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
    ) -> str:
        """Call Qwen AI and return text response."""
        self.status = AgentStatus.THINKING
        try:
            result = await self.pool.ask(
                prompt,
                system_prompt=system_prompt,
                history=history,
            )
            return result
        finally:
            self.status = AgentStatus.WORKING

    async def think_with_search(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
    ) -> str:
        """Call Qwen AI with web search enabled — gets real-time data."""
        self.status = AgentStatus.THINKING
        try:
            result = await self.pool.ask(
                prompt,
                system_prompt=system_prompt,
                history=history,
                web_search=True,
            )
            return result
        finally:
            self.status = AgentStatus.WORKING

    async def think_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
    ) -> dict:
        """Call Qwen AI and return parsed JSON response."""
        self.status = AgentStatus.THINKING
        try:
            result = await self.pool.ask_json(
                prompt,
                system_prompt=system_prompt,
                history=history,
            )
            return result
        finally:
            self.status = AgentStatus.WORKING

    # ─── Event Helpers ───────────────────────────────────────────────────────

    async def emit(
        self,
        event_type: str,
        payload: dict,
        target_agent: Optional[str] = None,
        priority: int = 5,
    ):
        """Emit an event from this agent."""
        await self.bus.emit(
            event_type=event_type,
            source_agent=self.agent_id,
            payload=payload,
            target_agent=target_agent,
            priority=priority,
        )

    async def notify_telegram(self, message: str, level: str = "info"):
        """Send a notification to the Telegram bot."""
        await self.emit(
            EventType.TELEGRAM_NOTIFY,
            payload={
                "message": message,
                "level": level,
                "agent": self.role,
            },
        )

    # ─── Error Handling ──────────────────────────────────────────────────────

    async def handle_error(self, error: Exception, context: str = ""):
        """Standard error handler — logs, updates status, emits event."""
        self.tasks_failed += 1
        self.status = AgentStatus.ERROR
        msg = f"[{self.role}] Error in {context}: {error}"
        self.logger.error(msg)
        await self.emit(
            EventType.AGENT_ERROR,
            payload={
                "agent_id": self.agent_id,
                "role": self.role,
                "error": str(error),
                "context": context,
            },
        )
        await self.notify_telegram(
            f"⚠️ **{self.role}** encountered an error\n`{str(error)[:200]}`",
            level="error"
        )

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "status": self.status,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "uptime_seconds": int(time.time() - self.created_at),
        }

    def __repr__(self) -> str:
        return f"<{self.role} id={self.agent_id[:8]} status={self.status}>"
