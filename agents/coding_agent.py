"""
agents/coding_agent.py — Coding Agent
Writes, fixes, and optimizes Python strategy scripts
"""
import logging
import os
import uuid
import time
import re

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from file_manager.manager import FileManager
from execution.package_manager import get_package_manager
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Python developer specializing in algorithmic trading systems.

You write clean, well-documented, production-quality Python code for:
- Trading strategy implementations
- Backtesting scripts
- Data processing pipelines

IMPORTANT — PACKAGE FREEDOM:
You are NOT limited to any pre-defined list of libraries.
The system has a dynamic package installer — any package you import
will be auto-installed via pip if missing.

You may freely use:
- pandas, numpy (always available)
- pandas_ta, ta, finta, vectorbt (TA libraries)
- scikit-learn, lightgbm, xgboost (ML)
- statsmodels, scipy, arch (statistics)
- stable_baselines3, gymnasium (reinforcement learning)
- optuna (hyperparameter optimization)
- pykalman, hmmlearn, ruptures (time series / regime detection)
- stumpy (matrix profile / pattern detection)
- ccxt (exchange data — though we use synthetic by default)
- OR ANY OTHER package you think is useful

Rules:
1. Handle edge cases (empty data, NaN values)
2. Add clear comments explaining the logic
3. Never use infinite loops
4. Always include a if __name__ == "__main__": block with JSON output
5. Write defensive code with proper error handling
6. Keep backtest self-contained with synthetic data (no external API calls)
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger and code fixer.

When given buggy code and error messages, you:
1. Analyze the exact error
2. Identify the root cause
3. Fix the code minimally and correctly
4. Preserve the original logic and intent
5. Return ONLY the complete fixed Python code, nothing else
"""


class CodingAgent(BaseAgent):

    ROLE = "CodingAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )
        self.file_manager = FileManager()
        self.pkg_manager = get_package_manager()

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.CODE_REQUESTED, self._on_code_requested)
        self.bus.subscribe(EventType.CODE_FIX_REQUESTED, self._on_fix_requested)

    async def _on_code_requested(self, event: Event):
        strategy = event.payload.get("strategy", {})
        await self.generate_strategy_code(strategy)

    async def _on_fix_requested(self, event: Event):
        script_path = event.payload.get("script_path", "")
        error_context = event.payload.get("error_context", "")
        strategy_id = event.payload.get("strategy_id", "")
        attempt = event.payload.get("attempt", 1)
        await self.fix_code(script_path, error_context, strategy_id, attempt)

    async def generate_strategy_code(self, strategy: dict) -> str:
        """Generate a Python backtesting script for a strategy."""
        self.status = AgentStatus.WORKING
        strategy_name = strategy.get("strategy_name", "Unknown")
        strategy_id = strategy.get("strategy_id", str(uuid.uuid4()))

        self.logger.info(f"[Coding] Generating code for: {strategy_name}")
        await self.notify_telegram(
            f"💻 **Coding Agent** writing strategy code\n"
            f"📝 Strategy: `{strategy_name}`",
            level="info"
        )

        prompt = f"""
Generate a complete Python backtesting script for this trading strategy:

STRATEGY: {strategy.get('strategy_name')}
OVERVIEW: {strategy.get('overview')}

Entry Conditions: {strategy.get('entry_conditions')}
Exit Conditions: {strategy.get('exit_conditions')}
Indicators: {strategy.get('indicators')}
Timeframes: {strategy.get('timeframes')}
Symbols: {strategy.get('symbols', ['BTCUSDT'])}
Stop Loss: {strategy.get('stop_loss_pct', 0.02) * 100}%
Take Profit: {strategy.get('take_profit_pct', 0.04) * 100}%

PACKAGE FREEDOM — You can import ANY Python library you want.
The system will auto-install missing packages before running.
Examples you can use freely:
- pandas_ta (for 130+ indicators)
- statsmodels (for statistical models)
- scikit-learn (for ML signals)
- lightgbm / xgboost (for ML-based entry signals)
- scipy (for signal processing / curve fitting)
- pykalman (for Kalman filter)
- hmmlearn (for Hidden Markov Models / regime detection)
- ruptures (for changepoint detection)
- optuna (for parameter optimization)
- OR any other library that fits the strategy

REQUIREMENTS:
1. Generate SYNTHETIC test data (500+ bars of OHLCV) — DO NOT use external APIs
2. Implement the full strategy logic using whatever libraries make sense
3. Produce signals: 1=long, -1=short, 0=flat
4. Calculate metrics: total_trades, winrate, pnl_pct, max_drawdown, profit_factor, sharpe_ratio
5. Print results as JSON: {{"strategy_id": "...", "metrics": {{...}}, "signals_generated": N}}
6. Handle all NaN values
7. Be creative — if ML would make this strategy better, use it

Strategy ID to embed: {strategy_id}
"""
        try:
            code = await self.think(prompt, system_prompt=SYSTEM_PROMPT)
            code = self._extract_code(code)

            if not code:
                raise ValueError("AI returned empty code")

            # Pre-scan and install any packages the AI decided to use
            pre_installed = await self.pkg_manager.install_from_imports(code)
            if pre_installed:
                self.logger.info(
                    f"[Coding] Pre-installed packages: {pre_installed}"
                )
                await self.notify_telegram(
                    f"📦 **Auto-installed** `{len(pre_installed)}` packages\n"
                    f"`{', '.join(pre_installed[:5])}`",
                    level="info"
                )

            # Save script
            filename = f"strategy_{strategy_id[:8]}_{int(time.time())}.py"
            script_path = self.file_manager.save_strategy_script(filename, code)

            # Save to DB
            store = get_store()
            await store.save_strategy({
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "script_path": script_path,
                "code": code,
                "status": "coded",
                **{k: v for k, v in strategy.items() if k != "code"},
            })
            await store.log_agent_action(
                self.agent_id, self.role, "generate_code",
                {"strategy_id": strategy_id, "script_path": script_path}
            )

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            self.logger.info(f"[Coding] Script saved: {script_path}")
            await self.notify_telegram(
                f"✅ **Code Generated**\n"
                f"📄 File: `{filename}`\n"
                f"📊 Strategy: `{strategy_name}`",
                level="success"
            )

            # Trigger execution/backtest
            await self.emit(
                EventType.CODE_GENERATED,
                payload={
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "script_path": script_path,
                    "strategy": strategy,
                }
            )

            return script_path

        except Exception as e:
            await self.handle_error(e, f"generate_code for {strategy_name}")
            self.status = AgentStatus.IDLE
            return ""

    async def fix_code(
        self,
        script_path: str,
        error_context: str,
        strategy_id: str,
        attempt: int = 1,
    ) -> str:
        """Fix a broken script based on error context."""
        if attempt > 3:
            self.logger.warning(f"[Coding] Max fix attempts reached for {script_path}")
            await self.notify_telegram(
                f"❌ **Code Fix Failed** (max attempts)\n`{os.path.basename(script_path)}`",
                level="error"
            )
            return ""

        self.status = AgentStatus.WORKING
        self.logger.info(f"[Coding] Fixing code (attempt {attempt}): {script_path}")
        await self.notify_telegram(
            f"🔧 **Fixing Code** (attempt {attempt}/3)\n"
            f"📄 `{os.path.basename(script_path)}`",
            level="warning"
        )

        try:
            original_code = self.file_manager.read_file(script_path)

            prompt = f"""
Fix this Python script. It has the following error:

{error_context}

ORIGINAL CODE:
```python
{original_code}
```

Return ONLY the complete fixed Python code. No explanations, no markdown, just pure Python code.
Start with the import statements.
"""
            fixed_code = await self.think(prompt, system_prompt=FIX_SYSTEM_PROMPT)
            fixed_code = self._extract_code(fixed_code)

            if not fixed_code:
                raise ValueError("Fix returned empty code")

            # Save fixed version
            fixed_path = script_path.replace(".py", f"_fix{attempt}.py")
            self.file_manager.save_file(fixed_path, fixed_code)

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

            # Retry backtest with fixed code
            await self.emit(
                EventType.CODE_FIXED,
                payload={
                    "strategy_id": strategy_id,
                    "script_path": fixed_path,
                    "attempt": attempt,
                    "original_path": script_path,
                }
            )

            return fixed_path

        except Exception as e:
            await self.handle_error(e, f"fix_code attempt {attempt}")
            self.status = AgentStatus.IDLE
            return ""

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from AI response."""
        text = text.strip()
        # Try ```python ... ``` block
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try ``` ... ``` block
        match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no markdown block, assume raw code
        if text.startswith("import") or text.startswith("#"):
            return text
        # Last resort: find first import statement
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("import") or line.strip().startswith("from"):
                return "\n".join(lines[i:])
        return text
