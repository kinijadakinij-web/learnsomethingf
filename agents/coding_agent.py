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

You write clean, well-documented, production-quality Python code for backtesting strategies.

AVAILABLE PACKAGES (auto-installed if missing):
- pandas, numpy (always available)
- pandas_ta, ta (TA indicators)
- scikit-learn, lightgbm, xgboost (ML)
- statsmodels, scipy (statistics)
- optuna (hyperparameter optimization)

Rules:
1. Handle edge cases (empty data, NaN values with dropna/fillna)
2. Generate SYNTHETIC OHLCV data (500+ bars) — NO external API calls
3. Always include: if __name__ == "__main__": block with JSON output
4. Calculate: total_trades, winrate, pnl_pct, max_drawdown, profit_factor, sharpe_ratio
5. Use CONSISTENT 4-space indentation — never mix tabs and spaces

CRITICAL OUTPUT FORMAT:
- Your ENTIRE response must be ONLY the code block below
- Start your response with: ```python
- End your response with: ```
- NO text before ```python, NO text after the closing ```
- NO explanations, NO comments outside the code block
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger.

CRITICAL OUTPUT FORMAT:
- Your ENTIRE response must be ONLY the fixed code block
- Start your response with: ```python
- End your response with: ```
- NO text before ```python, NO text after the closing ```
- NO explanations outside the code block

Fix the exact error given. Preserve original logic. Use 4-space indentation consistently.
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

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_syntax(code: str) -> str | None:
        """Return None if syntax OK, or a descriptive error string."""
        try:
            compile(code, "<generated>", "exec")
            return None
        except (SyntaxError, IndentationError) as e:
            return f"{type(e).__name__} at line {e.lineno}: {e.msg} — near {e.text!r}"

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from AI response.

        Tries all ```python blocks, picks the one that passes syntax check.
        Falls back to any ``` block, then raw text.
        """
        text = text.strip()

        # --- Collect ALL ```python ... ``` blocks ---
        python_blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if python_blocks:
            # Prefer the block that passes syntax validation
            for block in python_blocks:
                stripped = block.strip()
                if stripped and CodingAgent._validate_syntax(stripped) is None:
                    return stripped
            # None passed — return the longest for the fixer to work with
            return max(python_blocks, key=len).strip()

        # --- Any generic ``` block ---
        generic_blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
        if generic_blocks:
            for block in generic_blocks:
                stripped = block.strip()
                if stripped and CodingAgent._validate_syntax(stripped) is None:
                    return stripped
            return max(generic_blocks, key=len).strip()

        # --- Raw fallback: strip prose before first code line ---
        lines = text.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("import ", "from ", "def ", "class ", "#!")):
                return "\n".join(lines[i:])

        return text

    # ── Code generation ──────────────────────────────────────────────────────

    async def generate_strategy_code(self, strategy: dict) -> str:
        self.status = AgentStatus.WORKING
        strategy_name = strategy.get("strategy_name", "Unknown")
        strategy_id = strategy.get("strategy_id", str(uuid.uuid4()))

        self.logger.info(f"[Coding] Generating code for: {strategy_name}")
        await self.notify_telegram(
            f"💻 **Coding Agent** writing strategy code\n"
            f"📝 Strategy: `{strategy_name}`",
            level="info"
        )

        base_prompt = f"""
Write a complete Python backtesting script for this strategy.

STRATEGY: {strategy.get('strategy_name')}
OVERVIEW: {strategy.get('overview', 'N/A')}
Entry: {strategy.get('entry_long', strategy.get('entry_conditions', 'N/A'))}
Exit: {strategy.get('exit_long', strategy.get('exit_conditions', 'N/A'))}
Indicators: {strategy.get('indicators', [])}
Stop Loss: {float(strategy.get('stop_loss_pct', 0.02)) * 100:.1f}%
Take Profit: {float(strategy.get('take_profit_pct', 0.04)) * 100:.1f}%
Strategy ID: {strategy_id}

REQUIREMENTS:
- Synthetic OHLCV data (500 bars)
- Full signal logic (1=long, -1=short, 0=flat)
- Metrics: total_trades, winrate, pnl_pct, max_drawdown, profit_factor, sharpe_ratio
- Print JSON result at the end
- 4-space indentation, handle NaN with fillna/dropna

OUTPUT: respond with ONLY ```python ... ``` — nothing else.
"""

        try:
            code = ""
            syntax_err = "not generated yet"

            for attempt in range(3):
                if attempt == 0:
                    prompt = base_prompt
                else:
                    prompt = (
                        base_prompt +
                        f"\n\n[RETRY {attempt}/2] Previous code had error: {syntax_err}\n"
                        "Fix ALL indentation issues. Return ONLY ```python ... ```."
                    )

                raw = await self.think(prompt, system_prompt=SYSTEM_PROMPT)
                code = self._extract_code(raw)

                if not code:
                    syntax_err = "AI returned empty code"
                    continue

                syntax_err = self._validate_syntax(code)
                if syntax_err is None:
                    self.logger.info(f"[Coding] Valid code on attempt {attempt + 1}")
                    break
                self.logger.warning(f"[Coding] Syntax error attempt {attempt + 1}: {syntax_err}")

            if syntax_err is not None:
                raise ValueError(f"Code still has errors after 3 attempts: {syntax_err}")

            # Pre-install packages
            pre_installed = await self.pkg_manager.install_from_imports(code)
            if pre_installed:
                await self.notify_telegram(
                    f"📦 **Auto-installed** `{len(pre_installed)}` packages\n"
                    f"`{', '.join(pre_installed[:5])}`",
                    level="info"
                )

            # Save script
            filename = f"strategy_{strategy_id[:8]}_{int(time.time())}.py"
            script_path = self.file_manager.save_strategy_script(filename, code)

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

    # ── Code fixing ──────────────────────────────────────────────────────────

    async def fix_code(
        self,
        script_path: str,
        error_context: str,
        strategy_id: str,
        attempt: int = 1,
    ) -> str:
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

            prompt = f"""Fix this Python script.

ERROR:
{error_context}

BUGGY CODE:
```python
{original_code}
```

- Fix the exact error shown above
- Use 4-space indentation throughout
- Preserve original strategy logic
- OUTPUT: respond with ONLY ```python ... ``` — nothing else.
"""
            fixed_code = ""
            syntax_err = "not fixed yet"

            for fix_attempt in range(2):
                if fix_attempt > 0:
                    prompt += (
                        f"\n\n[RETRY] Fixed code still has error: {syntax_err}\n"
                        "Return ONLY ```python ... ``` with all syntax fixed."
                    )

                raw = await self.think(prompt, system_prompt=FIX_SYSTEM_PROMPT)
                fixed_code = self._extract_code(raw)

                if not fixed_code:
                    syntax_err = "empty code returned"
                    continue

                syntax_err = self._validate_syntax(fixed_code)
                if syntax_err is None:
                    break
                self.logger.warning(f"[Coding] Fix still broken: {syntax_err}")

            if not fixed_code:
                raise ValueError("Fix returned empty code")

            # Save fixed version (even if syntax check failed — let runner report real error)
            fixed_path = script_path.replace(".py", f"_fix{attempt}.py")
            self.file_manager.save_file(fixed_path, fixed_code)

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE

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
