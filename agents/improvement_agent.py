"""
agents/improvement_agent.py — Improvement Agent
Evolves strategies by mutating parameters and logic
"""
import logging
import uuid
import time

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert strategy optimizer and evolutionary algorithm designer.
You take existing trading strategies and improve them through:
- Parameter optimization
- Logic refinement
- Risk management improvements
- Indicator combinations
- Entry/exit condition enhancement

Your improvements must be specific, testable, and build on what worked.
"""


class ImprovementAgent(BaseAgent):
    ROLE = "ImprovementAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.IMPROVE_REQUESTED, self._on_improve_requested)

    async def _on_improve_requested(self, event: Event):
        strategy = event.payload.get("strategy", {})
        metrics = event.payload.get("metrics", {})
        evaluation = event.payload.get("evaluation", {})
        strategy_id = event.payload.get("strategy_id", "")
        await self.improve_strategy(strategy, metrics, evaluation, strategy_id)

    async def improve_strategy(
        self,
        strategy: dict,
        metrics: dict,
        evaluation: dict,
        parent_id: str,
    ) -> dict:
        self.status = AgentStatus.WORKING
        new_strategy_id = str(uuid.uuid4())
        strategy_name = strategy.get("strategy_name", "Unknown")

        self.logger.info(f"[Improvement] Improving: {strategy_name}")
        await self.notify_telegram(
            f"🧬 **Improvement Agent** evolving strategy\n"
            f"🔄 `{strategy_name}` → v{strategy.get('version', 1) + 1}",
            level="info"
        )

        prompt = f"""
You are evolving this trading strategy to a better version.

CURRENT STRATEGY: {strategy}
CURRENT METRICS: {metrics}
EVALUATION FEEDBACK: {evaluation}

Generate an IMPROVED version of this strategy.

Consider these improvement types:
1. Parameter tuning (adjust indicator periods, SL/TP percentages)
2. Add confirmation filter (add trend filter or volume condition)
3. Improve exits (trail stop, partial close)
4. Reduce false signals (add secondary indicator)
5. Optimize for the weaknesses identified in evaluation

Return the improved strategy as JSON (same structure as input strategy) with:
- All original fields preserved
- strategy_id: "{new_strategy_id}"
- version: {strategy.get('version', 1) + 1}
- parent_id: "{parent_id}"
- improvement_reason: "..." (what was changed and why)
- changes_made: [...] (list of specific changes)

Make the changes MEANINGFUL but CONSERVATIVE. Don't overhaul the whole strategy.
Focus on fixing the specific weaknesses.
"""
        try:
            improved = await self.think_json(prompt, system_prompt=SYSTEM_PROMPT)
            improved["strategy_id"] = new_strategy_id
            improved["parent_id"] = parent_id
            improved["version"] = strategy.get("version", 1) + 1
            improved["created_at"] = time.time()
            improved["status"] = "specified"

            store = get_store()
            await store.save_strategy(improved)
            await store.record_evolution(
                parent_id=parent_id,
                child_id=new_strategy_id,
                reason=improved.get("improvement_reason", "evolved"),
            )
            await store.log_agent_action(
                self.agent_id, self.role, "improve_strategy",
                {
                    "parent_id": parent_id,
                    "child_id": new_strategy_id,
                    "changes": improved.get("changes_made", []),
                }
            )

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            changes_str = "\n".join(
                f"  • {c}" for c in improved.get("changes_made", [])[:3]
            )
            await self.notify_telegram(
                f"🧬 **Strategy Evolved!**\n"
                f"📊 `{strategy_name}` → v{improved.get('version')}\n"
                f"🔄 Changes:\n{changes_str}",
                level="success"
            )

            # Send back to coding for implementation
            await self.emit(
                EventType.CODE_REQUESTED,
                payload={
                    "strategy": improved,
                    "strategy_id": new_strategy_id,
                }
            )

            await self.emit(
                EventType.IMPROVE_COMPLETED,
                payload={
                    "parent_id": parent_id,
                    "child_id": new_strategy_id,
                    "improved_strategy": improved,
                }
            )

            return improved

        except Exception as e:
            await self.handle_error(e, f"improve {strategy_name}")
            self.status = AgentStatus.IDLE
            return {}
