"""
agents/risk_agent.py — Risk Agent
Evaluates leverage risk, drawdown, position sizing before strategies go live
"""
import logging
import uuid

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store
import config

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    ROLE = "RiskAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.RISK_CHECK_REQUESTED, self._on_risk_check)
        self.bus.subscribe(EventType.BACKTEST_COMPLETED, self._on_backtest_completed)

    async def _on_backtest_completed(self, event: Event):
        """Automatically check risk after every backtest."""
        metrics = event.payload.get("metrics", {})
        strategy_id = event.payload.get("strategy_id", "")
        strategy = event.payload.get("strategy", {})
        await self.assess_risk(metrics, strategy_id, strategy)

    async def _on_risk_check(self, event: Event):
        metrics = event.payload.get("metrics", {})
        strategy_id = event.payload.get("strategy_id", "")
        strategy = event.payload.get("strategy", {})
        await self.assess_risk(metrics, strategy_id, strategy)

    async def assess_risk(self, metrics: dict, strategy_id: str, strategy: dict) -> dict:
        self.status = AgentStatus.WORKING

        leverage = int(strategy.get("leverage", 10))
        max_dd = float(metrics.get("max_drawdown_pct", 0))
        winrate = float(metrics.get("winrate", 0))
        liquidations = int(metrics.get("liquidations", 0))
        total_trades = int(metrics.get("total_trades", 1))

        # Rule-based risk scoring
        risk_flags = []
        risk_level = "low"

        if leverage > 20:
            risk_flags.append("Very high leverage (>20x) — extreme liquidation risk")
            risk_level = "critical"
        elif leverage > 10:
            risk_flags.append(f"High leverage ({leverage}x) — monitor closely")
            risk_level = "high"

        if max_dd > 0.5:
            risk_flags.append(f"Max drawdown {max_dd:.1%} — catastrophic risk")
            risk_level = "critical"
        elif max_dd > config.MAX_DRAWDOWN_THRESHOLD:
            risk_flags.append(f"Max drawdown {max_dd:.1%} exceeds threshold")
            if risk_level != "critical":
                risk_level = "high"

        liq_rate = liquidations / max(total_trades, 1)
        if liq_rate > 0.1:
            risk_flags.append(f"Liquidation rate {liq_rate:.1%} — dangerous")
            risk_level = "critical"

        if winrate < 0.35:
            risk_flags.append(f"Low winrate {winrate:.1%} — psychological risk")

        risk_ok = risk_level in ("low", "medium")

        risk_assessment = {
            "strategy_id": strategy_id,
            "risk_level": risk_level,
            "risk_ok": risk_ok,
            "risk_flags": risk_flags,
            "leverage_used": leverage,
            "max_drawdown_pct": max_dd,
            "liquidation_rate": liq_rate,
        }

        store = get_store()
        await store.save_strategy({
            "strategy_id": strategy_id,
            "risk_assessment": risk_assessment,
        })

        if risk_flags:
            flags_str = "\n".join(f"  ⚠️ {f}" for f in risk_flags)
            await self.notify_telegram(
                f"🛡️ **Risk Assessment**\n"
                f"📊 Strategy: `{strategy_id[:8]}`\n"
                f"🎯 Risk Level: **{risk_level.upper()}**\n"
                f"{flags_str}",
                level="warning" if not risk_ok else "info"
            )

        self.tasks_completed += 1
        self.status = AgentStatus.IDLE

        await self.emit(
            EventType.RISK_CHECK_COMPLETED,
            payload={"strategy_id": strategy_id, "risk_assessment": risk_assessment}
        )

        return risk_assessment
