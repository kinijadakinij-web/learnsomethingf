"""
agents/memory_agent.py — Memory Agent
Stores, retrieves, and manages institutional memory of the lab
"""
import logging
import uuid
import time

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)


class MemoryAgent(BaseAgent):
    ROLE = "MemoryAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )

    def _register_subscriptions(self):
        self.bus.subscribe("*", self._on_any_event)
        self.bus.subscribe(EventType.MEMORY_QUERY, self._on_memory_query)

    async def _on_memory_query(self, event: Event):
        """Handle memory retrieval requests."""
        query_type = event.payload.get("query_type", "")
        store = get_store()

        if query_type == "best_strategies":
            results = await store.get_best_strategies(limit=5)
            await self.emit(
                EventType.MEMORY_RESULT,
                payload={"query_type": query_type, "results": results},
                target_agent=event.source_agent,
            )
        elif query_type == "recent_research":
            results = await store.get_recent_research(limit=10)
            await self.emit(
                EventType.MEMORY_RESULT,
                payload={"query_type": query_type, "results": results},
                target_agent=event.source_agent,
            )
        elif query_type == "stats":
            stats = await store.get_stats()
            await self.emit(
                EventType.MEMORY_RESULT,
                payload={"query_type": query_type, "results": stats},
                target_agent=event.source_agent,
            )

    async def _on_any_event(self, event: Event):
        """Passively log important events to memory."""
        important_events = {
            EventType.STRATEGY_CREATED,
            EventType.BACKTEST_COMPLETED,
            EventType.EVALUATION_COMPLETED,
            EventType.IMPROVE_COMPLETED,
            EventType.BACKTEST_FAILED,
        }

        if event.event_type in important_events:
            store = get_store()
            await store.log_agent_action(
                agent_id=event.source_agent,
                role="MemoryAgent",
                action=f"logged_{event.event_type}",
                data={"payload_keys": list(event.payload.keys())},
            )

            # Store pattern memories for recurring insights
            if event.event_type == EventType.EVALUATION_COMPLETED:
                evaluation = event.payload.get("evaluation", {})
                if evaluation.get("assessment") in ("excellent", "good"):
                    strategy_id = event.payload.get("strategy_id", "")
                    await store.remember(
                        key=f"good_strategy_{strategy_id[:8]}",
                        value={
                            "strategy_id": strategy_id,
                            "assessment": evaluation.get("assessment"),
                            "strengths": evaluation.get("strengths", []),
                            "timestamp": time.time(),
                        },
                        category="successful_patterns"
                    )
