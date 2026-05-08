"""
core/event_bus.py — Async Event Bus for inter-agent communication
Agents publish events, other agents subscribe and react
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents a message/event between agents."""
    event_type: str
    source_agent: str
    target_agent: Optional[str]    # None = broadcast
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 5              # 1=highest, 10=lowest

    def __lt__(self, other: "Event") -> bool:
        return self.priority < other.priority


# ── Known Event Types ──────────────────────────────────────────────────────────
class EventType:
    # Lifecycle
    AGENT_STARTED        = "agent.started"
    AGENT_STOPPED        = "agent.stopped"
    AGENT_ERROR          = "agent.error"

    # Research cycle
    RESEARCH_REQUESTED   = "research.requested"
    RESEARCH_COMPLETED   = "research.completed"

    # Strategy cycle
    STRATEGY_REQUESTED   = "strategy.requested"
    STRATEGY_CREATED     = "strategy.created"

    # Coding cycle
    CODE_REQUESTED       = "code.requested"
    CODE_GENERATED       = "code.generated"
    CODE_FIX_REQUESTED   = "code.fix_requested"
    CODE_FIXED           = "code.fixed"

    # Backtest cycle
    BACKTEST_REQUESTED   = "backtest.requested"
    BACKTEST_COMPLETED   = "backtest.completed"
    BACKTEST_FAILED      = "backtest.failed"

    # Evaluation
    EVALUATION_REQUESTED = "evaluation.requested"
    EVALUATION_COMPLETED = "evaluation.completed"

    # Improvement
    IMPROVE_REQUESTED    = "improve.requested"
    IMPROVE_COMPLETED    = "improve.completed"

    # Risk
    RISK_CHECK_REQUESTED = "risk.check_requested"
    RISK_CHECK_COMPLETED = "risk.check_completed"

    # Memory
    MEMORY_STORE         = "memory.store"
    MEMORY_QUERY         = "memory.query"
    MEMORY_RESULT        = "memory.result"

    # Execution
    EXECUTE_REQUESTED    = "execute.requested"
    EXECUTE_COMPLETED    = "execute.completed"
    EXECUTE_FAILED       = "execute.failed"

    # System
    LOOP_TICK            = "system.loop_tick"
    SHUTDOWN             = "system.shutdown"
    TELEGRAM_NOTIFY      = "telegram.notify"


class EventBus:
    """
    Central async event bus.
    Agents subscribe to event types, events are dispatched asynchronously.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.PriorityQueue = None
        self._running = False
        self._history: List[Event] = []
        self._max_history = 500

    async def start(self):
        self._queue = asyncio.PriorityQueue()
        self._running = True
        asyncio.create_task(self._dispatcher())
        logger.info("[EventBus] Started")

    async def stop(self):
        self._running = False
        if self._queue:
            await self._queue.put((0, Event(
                event_type=EventType.SHUTDOWN,
                source_agent="bus",
                target_agent=None,
                payload={},
                priority=0
            )))

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe a handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"[EventBus] {handler.__qualname__} subscribed to {event_type}")

    async def publish(self, event: Event):
        """Publish an event to the bus."""
        if self._queue is None:
            logger.warning("[EventBus] Bus not started, dropping event")
            return
        await self._queue.put((event.priority, event))
        logger.debug(
            f"[EventBus] {event.source_agent} → {event.target_agent or 'ALL'} "
            f"[{event.event_type}]"
        )

    async def emit(
        self,
        event_type: str,
        source_agent: str,
        payload: dict,
        target_agent: Optional[str] = None,
        priority: int = 5,
    ):
        """Convenience method to create and publish an event."""
        event = Event(
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            payload=payload,
            priority=priority,
        )
        await self.publish(event)
        return event

    async def _dispatcher(self):
        """Main dispatch loop — routes events to subscribers."""
        while self._running:
            try:
                _, event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )

                if event.event_type == EventType.SHUTDOWN:
                    break

                # Store history
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

                # Find subscribers
                handlers = []
                handlers.extend(self._subscribers.get(event.event_type, []))
                handlers.extend(self._subscribers.get("*", []))  # wildcard

                # Filter by target if specified
                if event.target_agent:
                    handlers = [
                        h for h in handlers
                        if hasattr(h, "__self__")
                        and getattr(h.__self__, "agent_id", None) == event.target_agent
                        or not hasattr(h, "__self__")
                    ]

                # Dispatch to all handlers
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.create_task(handler(event))
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"[EventBus] Handler error: {e}")

                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[EventBus] Dispatch error: {e}")

    def recent_events(self, n: int = 20) -> List[Event]:
        return self._history[-n:]


# Global singleton
_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
