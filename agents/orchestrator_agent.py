"""
agents/orchestrator_agent.py — Orchestrator Agent
The brain that decides what happens next in the pipeline.
Monitors all agents, assigns priority tasks, detects stalls.
"""
import asyncio
import logging
import time
import uuid

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the master orchestrator of an autonomous AI trading research lab.
You have visibility into all agent activities, strategy performances, and lab health.
Your job is to make high-level decisions about what the lab should focus on next.
You think strategically about resource allocation and research priorities.
"""


class OrchestratorAgent(BaseAgent):
    """
    The strategic brain of the lab.
    
    Monitors all activity and intervenes when:
    - An agent is stuck
    - The lab is idle
    - A new research direction is needed
    - High-performing strategies need scaling
    """

    ROLE = "OrchestratorAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )
        self._last_activity = time.time()
        self._pipeline_in_progress = False
        self._pipeline_count = 0
        self._successful_strategies = 0
        self._failed_strategies = 0

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.LOOP_TICK, self._on_loop_tick)
        self.bus.subscribe(EventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
        self.bus.subscribe(EventType.IMPROVE_COMPLETED, self._on_improve_completed)
        self.bus.subscribe(EventType.BACKTEST_FAILED, self._on_backtest_failed)
        self.bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)

    async def _on_loop_tick(self, event: Event):
        """On each loop tick, decide what to do next."""
        if self._pipeline_in_progress:
            # Check for stall (pipeline running > 10 min)
            stall_time = time.time() - self._last_activity
            if stall_time > 600:
                self.logger.warning(
                    f"[Orchestrator] Pipeline stalled for {stall_time:.0f}s — resetting"
                )
                await self.notify_telegram(
                    f"⚠️ **Pipeline Stalled** — resetting after {stall_time:.0f}s",
                    level="warning"
                )
                self._pipeline_in_progress = False
        
        if not self._pipeline_in_progress:
            await self._start_new_pipeline()

    async def _start_new_pipeline(self):
        """Kick off a new research → strategy → code → backtest → evaluate pipeline."""
        self._pipeline_in_progress = True
        self._pipeline_count += 1
        self._last_activity = time.time()
        
        store = get_store()
        stats = await store.get_stats()

        self.logger.info(
            f"[Orchestrator] Starting pipeline #{self._pipeline_count} | "
            f"Strategies: {stats.get('total_strategies', 0)}"
        )

        await self.notify_telegram(
            f"🔄 **Pipeline #{self._pipeline_count} Started**\n"
            f"📊 Total strategies so far: `{stats.get('total_strategies', 0)}`\n"
            f"✅ Successful: `{self._successful_strategies}` | ❌ Failed: `{self._failed_strategies}`",
            level="system"
        )

        # Emit research request to kick off the chain
        await self.emit(
            EventType.RESEARCH_REQUESTED,
            payload={
                "pipeline_id": self._pipeline_count,
                "requested_by": self.agent_id,
            }
        )

    async def _on_evaluation_completed(self, event: Event):
        """Track completed evaluations."""
        self._last_activity = time.time()
        evaluation = event.payload.get("evaluation", {})
        assessment = evaluation.get("assessment", "unknown")
        
        if assessment in ("excellent", "good"):
            self._successful_strategies += 1
        
        # Mark pipeline as open for next cycle
        # (Improvement agent will continue if worth improving)
        if not evaluation.get("worth_improving"):
            self._pipeline_in_progress = False

    async def _on_improve_completed(self, event: Event):
        """When improvement finishes, a new pipeline auto-starts via CODE_REQUESTED."""
        self._last_activity = time.time()
        self._pipeline_in_progress = True  # Child pipeline now in progress

    async def _on_backtest_failed(self, event: Event):
        """Track failures, reset pipeline."""
        self._failed_strategies += 1
        self._pipeline_in_progress = False
        self._last_activity = time.time()

    async def _on_agent_error(self, event: Event):
        """Monitor agent errors, reset if needed."""
        self._last_activity = time.time()
        role = event.payload.get("role", "")
        
        # If a critical agent keeps failing, reset pipeline so it can retry fresh
        if role in ("CodingAgent", "ExecutionAgent"):
            self._pipeline_in_progress = False

    async def get_lab_summary(self) -> dict:
        """Generate a full lab summary report."""
        store = get_store()
        stats = await store.get_stats()
        best = await store.get_best_strategies(limit=3)
        recent_research = await store.get_recent_research(limit=5)

        prompt = f"""
You are the orchestrator of an autonomous AI trading research lab.
Generate a concise executive summary of the lab's progress.

Lab Stats: {stats}
Top Strategies: {[{'name': s.get('strategy_name'), 'score': s.get('score')} for s in best]}
Recent Research Topics: {[r.get('topic', '') for r in recent_research]}
Pipelines run: {self._pipeline_count}
Successful strategies: {self._successful_strategies}
Failed strategies: {self._failed_strategies}

Provide:
1. Overall progress assessment
2. Key findings so far
3. Recommended next focus areas
4. Any concerns or issues

Keep it concise — max 200 words.
"""
        try:
            summary = await self.think(prompt, system_prompt=SYSTEM_PROMPT)
            return {
                "summary": summary,
                "stats": stats,
                "pipeline_count": self._pipeline_count,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"[OrchestratorAgent] Summary error: {e}")
            return {"error": str(e)}
