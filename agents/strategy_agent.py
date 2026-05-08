"""
agents/strategy_agent.py — Strategy Agent
Converts research into concrete, testable strategy specifications
"""
import logging
import uuid
import time

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior quantitative trading strategy architect.

You take raw research ideas and convert them into precise, fully-specified trading strategies.
You define exact entry/exit rules, parameter values, risk management, and position sizing.
Your strategies must be implementable in Python code without ambiguity.
"""


class StrategyAgent(BaseAgent):
    ROLE = "StrategyAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.RESEARCH_COMPLETED, self._on_research_completed)
        self.bus.subscribe(EventType.STRATEGY_REQUESTED, self._on_strategy_requested)

    async def _on_research_completed(self, event: Event):
        research = event.payload.get("research", {})
        await self.create_strategy_from_research(research)

    async def _on_strategy_requested(self, event: Event):
        research = event.payload.get("research", {})
        await self.create_strategy_from_research(research)

    async def create_strategy_from_research(self, research: dict) -> dict:
        self.status = AgentStatus.WORKING
        strategy_name = (
            research.get("strategy_name")
            or research.get("name")
            or research.get("title")
            or research.get("topic", "")[:50]
            or "Unknown Strategy"
        )
        strategy_id = str(uuid.uuid4())

        self.logger.info(f"[Strategy] Building strategy: {strategy_name}")
        await self.notify_telegram(
            f"🏗️ **Strategy Agent** building strategy spec\n"
            f"📊 `{strategy_name}`",
            level="info"
        )

        prompt = f"""
Based on this research, create a precise, fully-specified trading strategy:

Research:
{research}

Define the EXACT strategy specification:
1. Exact indicator parameters (e.g., EMA(20), RSI(14, overbought=70))
2. Precise entry rules (e.g., "RSI crosses above 30 AND price > EMA20")
3. Precise exit rules
4. Stop loss calculation method
5. Take profit calculation method
6. Position sizing (fixed % of capital)
7. Maximum concurrent positions
8. Cooldown period between trades
9. Which symbol and timeframe to test first

Return a JSON object with EXACTLY these keys filled with real values (no placeholders):
{{
  "strategy_id": "{strategy_id}",
  "strategy_name": "{strategy_name}",
  "version": 1,
  "parent_id": null,
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "indicators": [
    {{"name": "EMA", "period": 20, "source": "close"}},
    {{"name": "RSI", "period": 14}}
  ],
  "entry_long": "EMA20 crosses above EMA50 AND RSI > 50 AND volume > 20-bar average",
  "entry_short": "EMA20 crosses below EMA50 AND RSI < 50 AND volume > 20-bar average",
  "exit_long": "EMA20 crosses below EMA50 OR RSI > 75 OR stop loss hit",
  "exit_short": "EMA20 crosses above EMA50 OR RSI < 25 OR stop loss hit",
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0.04,
  "position_size_pct": 0.95,
  "max_positions": 1,
  "leverage": 10,
  "cooldown_bars": 3,
  "description": "EMA crossover momentum strategy with RSI and volume confirmation on BTC 1h",
  "complexity": "simple",
  "expected_frequency": "2-4 trades per day",
  "risk_level": "medium"
}}

IMPORTANT: Replace ALL example values above with the actual strategy logic from the research. Do not copy the example verbatim.
"""
        try:
            strategy = await self.think_json(prompt, system_prompt=SYSTEM_PROMPT)
            strategy["strategy_id"] = strategy_id
            strategy["created_at"] = time.time()
            strategy["status"] = "specified"
            strategy["research_id"] = research.get("research_id", "")
            strategy["strategy_name"] = strategy_name

            # ── Normalize field names — Qwen kadang pakai key yang beda ──────
            # symbol
            strategy["symbol"] = (
                strategy.get("symbol") or strategy.get("pair") or
                strategy.get("ticker") or strategy.get("asset") or "BTCUSDT"
            )
            # timeframe
            strategy["timeframe"] = (
                strategy.get("timeframe") or strategy.get("interval") or
                strategy.get("tf") or strategy.get("time_frame") or "1h"
            )
            # leverage
            strategy["leverage"] = int(
                strategy.get("leverage") or strategy.get("lev") or 10
            )
            # stop_loss / take_profit
            strategy["stop_loss_pct"] = float(
                strategy.get("stop_loss_pct") or strategy.get("stop_loss") or
                strategy.get("sl_pct") or 0.02
            )
            strategy["take_profit_pct"] = float(
                strategy.get("take_profit_pct") or strategy.get("take_profit") or
                strategy.get("tp_pct") or 0.04
            )

            store = get_store()
            await store.save_strategy(strategy)
            await store.log_agent_action(
                self.agent_id, self.role, "create_strategy",
                {"strategy_id": strategy_id, "strategy_name": strategy_name}
            )

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            await self.notify_telegram(
                f"✅ **Strategy Specified**\n"
                f"🎯 `{strategy_name}`\n"
                f"📈 Symbol: `{strategy['symbol']}` | TF: `{strategy['timeframe']}`\n"
                f"⚡ Leverage: `{strategy['leverage']}x`",
                level="success"
            )

            # ── FIX: emit CODE_REQUESTED agar CodingAgent langsung pickup ────
            # (sebelumnya hanya emit STRATEGY_CREATED yang tidak didengar CodingAgent)
            await self.emit(
                EventType.STRATEGY_CREATED,
                payload={"strategy": strategy, "strategy_id": strategy_id}
            )
            await self.emit(
                EventType.CODE_REQUESTED,
                payload={"strategy": strategy, "strategy_id": strategy_id}
            )

            return strategy

        except Exception as e:
            await self.handle_error(e, f"create_strategy from {strategy_name}")
            self.status = AgentStatus.IDLE
            return {}
