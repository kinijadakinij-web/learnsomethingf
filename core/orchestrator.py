"""
core/orchestrator.py — Master Orchestrator
Controls the lab lifecycle, agent roster, and the autonomous evolution loop
"""
import asyncio
import logging
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import config
from ai.bearer_pool import BearerPool, get_pool
from core.event_bus import EventBus, EventType, Event, get_bus
from memory.mongodb_store import MongoStore, get_store
from telegram_bot.bot import TelegramNotifier, get_notifier

logger = logging.getLogger(__name__)
console = Console()


class Orchestrator:
    """
    The master controller of the Trading Lab.

    Responsibilities:
    1. Initialize all agents and services
    2. Manage the autonomous research loop
    3. Handle system-wide events (pause/resume/shutdown)
    4. Monitor agent health
    5. Log system-wide stats
    """

    def __init__(self):
        self.pool: BearerPool = get_pool()
        self.bus: EventBus = get_bus()
        self.store: MongoStore = get_store()
        self.notifier: TelegramNotifier = get_notifier()

        self._agents = {}
        self._running = False
        self._paused = False
        self._loop_count = 0
        self._start_time = time.time()

    async def initialize(self):
        """Initialize all services and agents."""
        console.print(Panel.fit(
            f"[bold cyan]🚀 {config.LAB_NAME}[/bold cyan]\n"
            "[dim]Autonomous AI Futures Trading Research Lab[/dim]",
            box=box.DOUBLE
        ))

        # Start event bus
        await self.bus.start()
        console.print("✅ [green]Event bus started[/green]")

        # Connect MongoDB
        await self.store.connect()
        console.print("✅ [green]MongoDB connected[/green]")

        # Register for system events
        self.bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)
        self.bus.subscribe("system.pause", self._on_pause)
        self.bus.subscribe("system.resume", self._on_resume)
        self.bus.subscribe(EventType.SHUTDOWN, self._on_shutdown)

        # Initialize agents
        await self._init_agents()

        # Start Telegram bot
        await self.notifier.start()
        console.print("✅ [green]Telegram bot started[/green]")

        console.print(f"\n[bold green]Lab initialized with {len(self._agents)} agents[/bold green]")
        self._print_agent_table()

    async def _init_agents(self):
        """Instantiate and start all specialized agents."""
        from agents.research_agent import ResearchAgent
        from agents.strategy_agent import StrategyAgent
        from agents.coding_agent import CodingAgent
        from agents.execution_agent import ExecutionAgent
        from agents.evaluation_agent import EvaluationAgent
        from agents.improvement_agent import ImprovementAgent
        from agents.memory_agent import MemoryAgent
        from agents.risk_agent import RiskAgent
        from agents.web_search_agent import WebSearchAgent

        agent_classes = [
            ResearchAgent,
            StrategyAgent,
            CodingAgent,
            ExecutionAgent,
            EvaluationAgent,
            ImprovementAgent,
            MemoryAgent,
            RiskAgent,
            WebSearchAgent,
        ]

        for cls in agent_classes:
            agent = cls()
            await agent.start()
            self._agents[cls.ROLE] = agent
            console.print(f"  🤖 [cyan]{cls.ROLE}[/cyan] started")

        await self.store.log_agent_action(
            "orchestrator", "Orchestrator", "init_complete",
            {"agent_count": len(self._agents)}
        )

    def _print_agent_table(self):
        table = Table(title="Active Agents", box=box.ROUNDED)
        table.add_column("Role", style="cyan")
        table.add_column("Agent ID", style="dim")
        table.add_column("Status", style="green")

        for role, agent in self._agents.items():
            table.add_row(role, agent.agent_id[:8], agent.status)

        console.print(table)

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def run(self):
        """Main autonomous research loop."""
        self._running = True
        logger.info("[Orchestrator] Starting main loop")

        await self.bus.emit(
            EventType.TELEGRAM_NOTIFY,
            source_agent="orchestrator",
            payload={
                "message": f"🔄 **Evolution Loop Starting**\nInterval: `{config.LOOP_INTERVAL_SECONDS}s`",
                "level": "system",
            }
        )

        while self._running:
            if self._paused:
                await asyncio.sleep(5)
                continue

            self._loop_count += 1
            loop_start = time.time()

            console.print(f"\n[bold blue]━━━ Loop #{self._loop_count} ━━━[/bold blue]")

            try:
                # Emit loop tick — Research Agent will pick this up
                await self.bus.emit(
                    EventType.LOOP_TICK,
                    source_agent="orchestrator",
                    payload={
                        "loop_count": self._loop_count,
                        "timestamp": time.time(),
                    }
                )

                # Log stats periodically
                if self._loop_count % 5 == 0:
                    await self._log_stats()

                # Wait for the interval
                elapsed = time.time() - loop_start
                wait_time = max(0, config.LOOP_INTERVAL_SECONDS - elapsed)

                console.print(
                    f"[dim]Loop #{self._loop_count} dispatched. "
                    f"Next tick in {wait_time:.0f}s[/dim]"
                )

                await asyncio.sleep(wait_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Orchestrator] Loop error: {e}")
                await asyncio.sleep(10)

    async def _log_stats(self):
        stats = await self.store.get_stats()
        uptime = int(time.time() - self._start_time)

        table = Table(title=f"Lab Stats (Loop #{self._loop_count})", box=box.SIMPLE)
        table.add_column("Metric")
        table.add_column("Value", style="cyan")

        table.add_row("Strategies", str(stats.get("total_strategies", 0)))
        table.add_row("Backtests", str(stats.get("total_backtests", 0)))
        table.add_row("Research", str(stats.get("total_research", 0)))
        table.add_row("Evolution Nodes", str(stats.get("total_evolution_nodes", 0)))
        table.add_row("Uptime", f"{uptime // 3600}h {(uptime % 3600) // 60}m")

        console.print(table)

        # Also send to Telegram every 5 loops
        await self.bus.emit(
            EventType.TELEGRAM_NOTIFY,
            source_agent="orchestrator",
            payload={
                "message": (
                    f"📊 **Lab Stats** (Loop #{self._loop_count})\n"
                    f"🔬 Strategies: `{stats.get('total_strategies', 0)}`\n"
                    f"📈 Backtests: `{stats.get('total_backtests', 0)}`\n"
                    f"🧪 Research: `{stats.get('total_research', 0)}`\n"
                    f"🧬 Evolution: `{stats.get('total_evolution_nodes', 0)}`"
                ),
                "level": "info",
            }
        )

    # ─── Event Handlers ───────────────────────────────────────────────────────

    async def _on_agent_error(self, event: Event):
        role = event.payload.get("role", "unknown")
        error = event.payload.get("error", "")
        logger.error(f"[Orchestrator] Agent error from {role}: {error}")

        # Attempt agent restart if critical
        if role in self._agents:
            agent = self._agents[role]
            if agent.tasks_failed > 10:
                logger.warning(f"[Orchestrator] Agent {role} has too many failures, restarting")
                await agent.stop()
                cls = type(agent)
                new_agent = cls()
                await new_agent.start()
                self._agents[role] = new_agent
                logger.info(f"[Orchestrator] Agent {role} restarted")

    async def _on_pause(self, event: Event):
        self._paused = True
        logger.info("[Orchestrator] Lab PAUSED")

    async def _on_resume(self, event: Event):
        self._paused = False
        logger.info("[Orchestrator] Lab RESUMED")

    async def _on_shutdown(self, event: Event):
        self._running = False

    # ─── Shutdown ─────────────────────────────────────────────────────────────

    async def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        logger.info("[Orchestrator] Shutting down...")

        for role, agent in self._agents.items():
            await agent.stop()
            logger.info(f"  Stopped: {role}")

        await self.notifier.stop()
        await self.bus.stop()
        await self.store.disconnect()

        console.print("\n[bold red]Lab shutdown complete.[/bold red]")
