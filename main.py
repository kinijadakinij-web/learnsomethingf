"""
main.py — Entry point for the Autonomous Trading Lab
Run: python main.py
"""
import asyncio
import logging
import os
import signal
import sys

from rich.console import Console
from rich.logging import RichHandler

import config

# ── Buat semua direktori DULU sebelum apapun ──────────────────────────────────
# FileHandler akan crash kalau folder belum ada
for _dir in [
    config.GENERATED_DIR,
    config.STRATEGIES_DIR,
    config.REPORTS_DIR,
    config.BACKTESTS_DIR,
    config.LOGS_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# ── Logging setup (setelah folder dibuat) ─────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler(f"{config.LOGS_DIR}/lab.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
console = Console()


async def main():
    from core.orchestrator import Orchestrator

    orchestrator = Orchestrator()

    loop = asyncio.get_event_loop()

    def _shutdown_handler():
        console.print("\n[yellow]Shutdown signal received...[/yellow]")
        asyncio.create_task(orchestrator.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_handler)
        except NotImplementedError:
            pass  # Windows

    try:
        await orchestrator.initialize()
        await orchestrator.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]KeyboardInterrupt — shutting down...[/yellow]")
    except Exception as e:
        logger.exception(f"[Main] Fatal error: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
