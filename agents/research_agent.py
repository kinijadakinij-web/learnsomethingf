"""
agents/research_agent.py — Research Agent
Autonomously researches trading ideas, indicators, market conditions
"""
import logging
import time
import uuid
from typing import Optional

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior quantitative trading researcher specializing in crypto futures markets.

Your job is to generate novel, testable trading strategy ideas based on:
- Technical analysis (indicators, price action, patterns)
- Market microstructure (volume, order flow, funding rates)
- Statistical relationships and mean reversion
- Momentum and trend following
- Volatility patterns

Focus on FUTURES markets (perpetual swaps). Consider leverage and liquidation risks.
Always think about what makes a strategy robust and generalizable.
"""


class ResearchAgent(BaseAgent):
    """
    Generates trading research ideas autonomously.
    Pulls from memory to avoid repeated ideas.
    Sends research to StrategyAgent.
    """

    ROLE = "ResearchAgent"

    RESEARCH_TOPICS = [
        "momentum strategies on BTC/USDT perpetual futures using volume confirmation",
        "mean reversion strategies on altcoin futures during high volatility periods",
        "funding rate arbitrage and long/short bias strategies",
        "breakout strategies using Bollinger Bands and ATR on ETH futures",
        "RSI divergence strategies on 4h timeframe for BTC perpetuals",
        "VWAP deviation strategies for intraday crypto futures",
        "multi-timeframe trend following with dynamic position sizing",
        "volatility regime switching strategies for crypto futures",
        "order book imbalance strategies using market depth",
        "EMA crossover systems with adaptive parameters",
        "Keltner Channel breakout strategies with volume filter",
        "Supertrend indicator strategies with risk management",
        "MACD histogram strategies with trend filter",
        "Stochastic RSI overbought/oversold with trend confirmation",
    ]

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )
        self._topic_idx = 0

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.RESEARCH_REQUESTED, self._on_research_requested)
        self.bus.subscribe(EventType.LOOP_TICK, self._on_loop_tick)
        self.bus.subscribe("web_search.completed", self._on_web_intel_received)

    async def _on_web_intel_received(self, event: Event):
        """When WebSearchAgent sends intel, use it to enrich next research."""
        intel = event.payload.get("intel", {})
        strategy_ideas = intel.get("strategy_ideas", [])
        if strategy_ideas and self.status == AgentStatus.IDLE:
            # Use the web-found idea as research topic
            best_idea = strategy_ideas[0]
            topic = best_idea.get("concept") or best_idea.get("name") or ""
            if topic:
                self.logger.info(f"[Research] Web intel triggered research: {topic}")
                await self.conduct_research(topic=topic, web_context=intel)

    async def _on_loop_tick(self, event: Event):
        """Proactively generate research on each loop tick."""
        if self.status == AgentStatus.IDLE and self._running:
            await self.conduct_research()

    async def _on_research_requested(self, event: Event):
        """Handle explicit research requests."""
        topic = event.payload.get("topic", "")
        web_context = event.payload.get("web_context", None)
        await self.conduct_research(topic=topic, web_context=web_context)

    async def conduct_research(
        self,
        topic: Optional[str] = None,
        web_context: Optional[dict] = None,
    ) -> dict:
        """Core research function — generates a trading idea and sends to strategy agent."""
        self.status = AgentStatus.WORKING

        # Pick topic (rotate through list or use provided)
        if not topic:
            topic = self.RESEARCH_TOPICS[self._topic_idx % len(self.RESEARCH_TOPICS)]
            self._topic_idx += 1

        self.logger.info(f"[Research] Researching: {topic}")
        await self.notify_telegram(
            f"🔬 **Research Agent** starting new research\n📌 Topic: _{topic}_",
            level="info"
        )

        # Check if we've researched this recently
        store = get_store()
        past = await store.get_recent_research(limit=50)
        past_topics = [r.get("topic", "") for r in past]

        # Generate research with AI
        web_context_section = ""
        if web_context:
            web_context_section = f"""
REAL-TIME WEB INTELLIGENCE (use this to make research more current):
- Market Sentiment: {web_context.get('sentiment', 'unknown')}
- Market Conditions: {web_context.get('market_conditions', '')}
- Trending Indicators: {web_context.get('indicators_trending', [])}
- Key Findings from Web: {web_context.get('key_findings', [])[:3]}
- Volatility Outlook: {web_context.get('volatility_outlook', 'normal')}
"""

        prompt = f"""
Research this trading strategy concept for crypto futures:
**{topic}**
{web_context_section}
Provide a detailed research report with:
1. Strategy overview and core logic
2. Entry and exit conditions (specific and testable)
3. Indicators needed (name, parameters)
4. Timeframes best suited
5. Risk management approach
6. Expected characteristics (winrate range, typical holding period)
7. Potential weaknesses and edge cases
8. Symbols to test (suggest 2-3 crypto futures pairs)
9. Unique insights that make this strategy edge

Past topics already researched (avoid overlap):
{', '.join(past_topics[-10:])}

Return a JSON object with EXACTLY these keys filled with real values (no placeholders):
{{
  "strategy_name": "EMA Momentum Volume Filter",
  "topic": "momentum strategies on BTC/USDT using volume confirmation",
  "overview": "Uses EMA crossover confirmed by volume spike to enter momentum trades",
  "entry_conditions": ["EMA20 crosses above EMA50", "Volume > 1.5x 20-period average"],
  "exit_conditions": ["EMA20 crosses below EMA50", "RSI above 75"],
  "indicators": [{{"name": "EMA", "params": {{"period": 20}}}}, {{"name": "EMA", "params": {{"period": 50}}}}],
  "timeframes": ["1h", "4h"],
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0.04,
  "expected_winrate": 0.55,
  "holding_period": "4-24 hours",
  "weaknesses": ["Choppy markets cause whipsaws", "Lagging indicator in fast moves"],
  "unique_edge": "Volume confirmation reduces false breakouts by 30 percent",
  "complexity": "simple"
}}

IMPORTANT: Replace ALL example values above with real values specific to the strategy you are describing. Do not copy the example verbatim.
"""
        try:
            research = await self.think_json(prompt, system_prompt=SYSTEM_PROMPT)

            # Fallback: kalau strategy_name kosong, generate dari topic
            if not research.get("strategy_name"):
                # Coba key alternatif yang mungkin dipakai Qwen
                research["strategy_name"] = (
                    research.get("name")
                    or research.get("title")
                    or research.get("strategy")
                    or " ".join(w.capitalize() for w in topic.split()[:5])
                )

            research["topic"] = topic
            research["researched_by"] = self.agent_id
            research["timestamp"] = time.time()

            # Save to MongoDB
            rid = await store.save_research(research)
            research["research_id"] = rid

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            self.logger.info(
                f"[Research] Completed: {research.get('strategy_name', 'Unknown')}"
            )

            await self.notify_telegram(
                f"✅ **Research Complete**\n"
                f"📊 Strategy: `{research.get('strategy_name', topic)}`\n"
                f"🎯 Edge: {research.get('unique_edge', '')[:100]}",
                level="success"
            )

            # Send to StrategyAgent
            await self.emit(
                EventType.RESEARCH_COMPLETED,
                payload={
                    "research": research,
                    "research_id": rid,
                },
            )

            return research

        except Exception as e:
            await self.handle_error(e, f"research on '{topic}'")
            self.status = AgentStatus.IDLE
            return {}
