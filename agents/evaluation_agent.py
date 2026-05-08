"""
agents/evaluation_agent.py — Evaluation Agent
Evaluates backtest results, scores strategies, detects weaknesses
"""
import logging
import uuid
import time

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert quantitative strategy evaluator.
Analyze backtest results critically. Detect overfitting, instability, poor risk management.
Provide actionable improvement suggestions.
"""


class EvaluationAgent(BaseAgent):
    ROLE = "EvaluationAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.BACKTEST_COMPLETED, self._on_backtest_completed)
        self.bus.subscribe(EventType.BACKTEST_FAILED, self._on_backtest_failed)

    async def _on_backtest_failed(self, event: Event):
        strategy_id = event.payload.get("strategy_id", "")
        error = event.payload.get("error", "")
        store = get_store()
        await store.save_strategy({
            "strategy_id": strategy_id,
            "status": "failed",
            "failure_reason": error,
            "updated_at": time.time(),
        })
        await self.notify_telegram(
            f"🚫 **Strategy Failed** — archived\n`{strategy_id[:8]}`",
            level="error"
        )

    async def _on_backtest_completed(self, event: Event):
        metrics = event.payload.get("metrics", {})
        strategy_id = event.payload.get("strategy_id", "")
        strategy = event.payload.get("strategy", {})
        await self.evaluate(metrics, strategy_id, strategy)

    async def evaluate(self, metrics: dict, strategy_id: str, strategy: dict) -> dict:
        self.status = AgentStatus.WORKING
        self.logger.info(f"[Evaluation] Evaluating strategy: {strategy_id[:8]}")

        try:
            # Quick rule-based checks
            winrate = float(metrics.get("winrate", 0))
            profit_factor = float(metrics.get("profit_factor", 0))
            max_dd = float(metrics.get("max_drawdown", 1))
            score = float(metrics.get("score", 0))
            total_trades = int(metrics.get("total_trades", 0))

            is_promising = (
                winrate >= config.MIN_WINRATE_THRESHOLD
                and profit_factor >= config.MIN_PROFIT_FACTOR
                and max_dd <= config.MAX_DRAWDOWN_THRESHOLD
                and total_trades >= 10
            )

            # AI deep evaluation
            prompt = f"""
Evaluate this trading strategy backtest result:

Strategy: {strategy.get('strategy_name', 'Unknown')}
Metrics: {metrics}
Config: leverage={strategy.get('leverage', 10)}x, 
        SL={strategy.get('stop_loss_pct', 0.02)*100}%, 
        TP={strategy.get('take_profit_pct', 0.04)*100}%

Provide:
1. Overall assessment (excellent/good/mediocre/poor)
2. Key strengths
3. Key weaknesses
4. Specific improvement suggestions (3-5 actionable items)
5. Risk concerns
6. Is this worth improving? (yes/no/maybe)
7. Suggested parameter changes

Return as JSON:
{{
  "assessment": "good",
  "strengths": [...],
  "weaknesses": [...],
  "improvements": [
    {{"type": "parameter", "description": "...", "specifics": {{...}}}},
    ...
  ],
  "risk_concerns": [...],
  "worth_improving": true,
  "parameter_suggestions": {{...}},
  "improvement_priority": "high|medium|low",
  "overall_score": 0.0
}}
"""
            evaluation = await self.think_json(prompt, system_prompt=SYSTEM_PROMPT)
            evaluation["strategy_id"] = strategy_id
            evaluation["evaluated_at"] = time.time()
            evaluation["rule_based_pass"] = is_promising

            store = get_store()
            await store.save_strategy({
                "strategy_id": strategy_id,
                "status": "evaluated",
                "is_promising": is_promising,
                "evaluation": evaluation,
                "metrics": metrics,
                "score": score,
                "updated_at": time.time(),
            })

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            grade_emoji = {"excellent": "🌟", "good": "✅", "mediocre": "⚠️", "poor": "❌"}.get(
                evaluation.get("assessment", "poor"), "❓"
            )

            await self.notify_telegram(
                f"{grade_emoji} **Strategy Evaluated**\n"
                f"📊 `{strategy.get('strategy_name', strategy_id[:8])}`\n"
                f"📈 WR: `{winrate:.1%}` | PF: `{profit_factor:.2f}` | DD: `{max_dd:.1%}`\n"
                f"🏆 Assessment: **{evaluation.get('assessment', 'N/A')}**\n"
                f"🔧 Worth Improving: `{evaluation.get('worth_improving')}`",
                level="info"
            )

            # Route to improvement or archive
            if evaluation.get("worth_improving") and is_promising:
                await self.emit(
                    EventType.IMPROVE_REQUESTED,
                    payload={
                        "strategy_id": strategy_id,
                        "strategy": strategy,
                        "metrics": metrics,
                        "evaluation": evaluation,
                    }
                )
            elif evaluation.get("worth_improving"):
                # Try improvement even if not fully passing thresholds
                await self.emit(
                    EventType.IMPROVE_REQUESTED,
                    payload={
                        "strategy_id": strategy_id,
                        "strategy": strategy,
                        "metrics": metrics,
                        "evaluation": evaluation,
                    }
                )

            await self.emit(
                EventType.EVALUATION_COMPLETED,
                payload={
                    "strategy_id": strategy_id,
                    "evaluation": evaluation,
                    "metrics": metrics,
                }
            )

            return evaluation

        except Exception as e:
            await self.handle_error(e, f"evaluate {strategy_id}")
            self.status = AgentStatus.IDLE
            return {}
