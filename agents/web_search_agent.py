"""
agents/web_search_agent.py — Web Search Agent
Searches the web for real-time market intelligence, new strategies,
news events, and quant research — feeds findings to ResearchAgent.

Uses Qwen's built-in web search (auto_search: True) so no extra API needed.
"""
import logging
import time
import uuid
from typing import Optional

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional crypto market intelligence analyst.

You search the web to find:
- Current crypto market conditions and sentiment
- New quantitative trading techniques and strategies
- Recent academic research on algorithmic trading
- News events that could affect crypto volatility
- Latest indicator techniques and improvements
- What professional quant traders are discussing

You synthesize web search results into actionable intelligence
that can improve trading strategy research and development.

Always cite what you found and be specific about dates and sources.
"""

# Search queries that rotate each cycle
SEARCH_QUERIES = [
    # Market intelligence
    "crypto futures trading strategy 2025 profitable",
    "bitcoin BTC volatility patterns analysis 2025",
    "ethereum futures technical analysis strategies",
    "crypto perpetual swap funding rate strategy",

    # Quant techniques
    "quantitative trading crypto new indicators 2025",
    "momentum strategy crypto futures backtesting",
    "mean reversion crypto trading algorithmic 2025",
    "machine learning crypto price prediction strategy",

    # Market structure
    "crypto market microstructure order flow trading",
    "bitcoin whale activity on-chain trading signal",
    "crypto liquidation cascade trading strategy",
    "funding rate arbitrage perpetual futures 2025",

    # Academic / research
    "algo trading crypto research paper 2025",
    "risk management crypto leverage futures strategy",
    "crypto volume profile VWAP trading technique",
    "breakout strategy crypto futures backtest 2025",
]


class WebSearchAgent(BaseAgent):
    """
    Searches the web for real-time trading intelligence.

    Pipeline:
    1. Pick a search topic (rotates through SEARCH_QUERIES)
    2. Ask Qwen with web_search=True
    3. Extract structured intelligence from results
    4. Save to MongoDB memory bank
    5. Emit to ResearchAgent as enriched context

    Triggers:
    - LOOP_TICK (every N loops, not every single one)
    - SEARCH_REQUESTED events
    """

    ROLE = "WebSearchAgent"
    # Only search every 3rd loop tick to not spam the API
    SEARCH_EVERY_N_LOOPS = 3

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )
        self._query_idx = 0
        self._loop_counter = 0
        self._last_search_time = 0.0

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.LOOP_TICK, self._on_loop_tick)
        self.bus.subscribe("search.requested", self._on_search_requested)

    async def _on_loop_tick(self, event: Event):
        self._loop_counter += 1
        # Only run every N loops AND at least 5 minutes since last search
        enough_time = (time.time() - self._last_search_time) > 300
        right_loop = (self._loop_counter % self.SEARCH_EVERY_N_LOOPS == 0)

        if right_loop and enough_time and self.status == AgentStatus.IDLE:
            await self.search_and_report()

    async def _on_search_requested(self, event: Event):
        query = event.payload.get("query", "")
        topic = event.payload.get("topic", "")
        await self.search_and_report(query=query, topic=topic)

    async def search_and_report(
        self,
        query: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> dict:
        """
        Core search function.
        Searches the web and extracts actionable trading intelligence.
        """
        self.status = AgentStatus.WORKING
        self._last_search_time = time.time()

        # Pick query
        if not query:
            query = SEARCH_QUERIES[self._query_idx % len(SEARCH_QUERIES)]
            self._query_idx += 1

        self.logger.info(f"[WebSearch] Searching: {query}")
        await self.notify_telegram(
            f"🌐 **Web Search Agent** scanning the web\n"
            f"🔍 Query: _{query}_",
            level="info"
        )

        # ── Phase 1: Raw web search ────────────────────────────────────────────
        search_prompt = f"""
Search the web for: "{query}"

Find the most relevant, recent, and actionable information about this topic
for crypto futures trading strategy development.

Look for:
1. Current market conditions or trends
2. Specific trading techniques or strategies mentioned
3. Indicators or tools being used
4. Risk factors or warnings
5. Any quantitative insights (numbers, percentages, timeframes)
6. Links or sources worth noting

Summarize what you found in detail.
"""
        try:
            raw_findings = await self.think_with_search(
                search_prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            # ── Phase 2: Structure the findings ───────────────────────────────
            structure_prompt = f"""
Based on these web search findings about "{query}":

---
{raw_findings}
---

Extract and structure this into actionable trading intelligence.

Return JSON:
{{
  "query": "{query}",
  "search_date": "{time.strftime('%Y-%m-%d')}",
  "market_conditions": "...",
  "key_findings": [
    "finding 1",
    "finding 2",
    "..."
  ],
  "strategy_ideas": [
    {{
      "name": "...",
      "concept": "...",
      "why_now": "...",
      "indicators_mentioned": [...],
      "timeframes_mentioned": [...]
    }}
  ],
  "risk_warnings": [...],
  "indicators_trending": [...],
  "sentiment": "bullish|bearish|neutral|mixed",
  "volatility_outlook": "high|normal|low",
  "actionable_for_research": true,
  "priority": "high|medium|low",
  "summary": "2-3 sentence summary"
}}
"""
            store = get_store()

            try:
                intel = await self.think_json(
                    structure_prompt,
                    system_prompt=SYSTEM_PROMPT,
                )
            except Exception:
                # If JSON parsing fails, store raw findings anyway
                intel = {
                    "query": query,
                    "search_date": time.strftime("%Y-%m-%d"),
                    "raw_findings": raw_findings[:2000],
                    "key_findings": [raw_findings[:500]],
                    "strategy_ideas": [],
                    "actionable_for_research": True,
                    "priority": "medium",
                    "summary": raw_findings[:200],
                }

            intel["agent_id"] = self.agent_id
            intel["timestamp"] = time.time()
            intel["search_type"] = "web"

            # Save to memory bank
            await store.remember(
                key=f"web_intel_{int(time.time())}",
                value=intel,
                category="web_intelligence",
            )

            # Also save as research if actionable
            if intel.get("actionable_for_research"):
                await store.save_research({
                    "topic": f"[WEB] {query}",
                    "source": "web_search",
                    "strategy_name": f"Web-Researched: {query[:40]}",
                    "overview": intel.get("summary", ""),
                    "key_findings": intel.get("key_findings", []),
                    "indicators": [
                        {"name": ind} for ind in intel.get("indicators_trending", [])
                    ],
                    "market_conditions": intel.get("market_conditions", ""),
                    "sentiment": intel.get("sentiment", "neutral"),
                    "researched_by": self.agent_id,
                    "timestamp": time.time(),
                })

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            # Build Telegram notification
            findings_preview = "\n".join(
                f"  • {f[:80]}" for f in intel.get("key_findings", [])[:3]
            )
            strategy_count = len(intel.get("strategy_ideas", []))

            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                intel.get("priority", "medium"), "🟡"
            )
            sentiment_emoji = {
                "bullish": "📈", "bearish": "📉", "neutral": "➡️", "mixed": "↕️"
            }.get(intel.get("sentiment", "neutral"), "➡️")

            await self.notify_telegram(
                f"🌐 **Web Intelligence Report**\n"
                f"🔍 Query: _{query[:50]}_\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{sentiment_emoji} Sentiment: `{intel.get('sentiment', 'N/A')}`\n"
                f"{priority_emoji} Priority: `{intel.get('priority', 'N/A')}`\n"
                f"💡 Strategy ideas found: `{strategy_count}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📋 Key Findings:\n{findings_preview}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📝 {intel.get('summary', '')[:150]}",
                level="success"
            )

            # If high priority + has strategy ideas → trigger research immediately
            if intel.get("priority") == "high" and intel.get("strategy_ideas"):
                best_idea = intel["strategy_ideas"][0]
                await self.emit(
                    EventType.RESEARCH_REQUESTED,
                    payload={
                        "topic": best_idea.get("concept", query),
                        "web_context": intel,
                        "source": "web_search_agent",
                    },
                    priority=2,  # High priority
                )
                self.logger.info(
                    f"[WebSearch] High-priority idea → triggering research: "
                    f"{best_idea.get('name', '')}"
                )
            else:
                # Normal priority → emit as context for next research cycle
                await self.emit(
                    "web_search.completed",
                    payload={
                        "intel": intel,
                        "query": query,
                    }
                )

            return intel

        except Exception as e:
            await self.handle_error(e, f"web_search '{query}'")
            self.status = AgentStatus.IDLE
            return {}

    async def search_market_conditions(self) -> dict:
        """
        Targeted search for current market conditions.
        Called by OrchestratorAgent before starting a new pipeline.
        """
        query = f"bitcoin ethereum crypto market analysis {time.strftime('%B %Y')}"
        return await self.search_and_report(query=query)

    async def search_for_indicator(self, indicator_name: str) -> dict:
        """
        Search for specific indicator usage in crypto trading.
        Called by ResearchAgent when it wants more info on an indicator.
        """
        query = f"{indicator_name} indicator crypto futures trading strategy"
        return await self.search_and_report(query=query)

    async def search_recent_quant_research(self) -> dict:
        """Search for recent quant trading papers and techniques."""
        query = f"quantitative trading crypto research {time.strftime('%Y')}"
        return await self.search_and_report(query=query)
