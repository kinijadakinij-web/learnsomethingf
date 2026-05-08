"""
agents/execution_agent.py — Execution Agent
Runs scripts, captures output, triggers fix loops on errors
"""
import json
import logging
import uuid

from core.agent_base import BaseAgent, AgentStatus
from core.event_bus import EventType, Event
from execution.runner import ScriptRunner
from execution.package_manager import get_package_manager
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    ROLE = "ExecutionAgent"

    def __init__(self, **kwargs):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            role=self.ROLE,
            **kwargs,
        )
        self.runner = ScriptRunner()
        self.pkg_manager = get_package_manager()

    def _register_subscriptions(self):
        self.bus.subscribe(EventType.CODE_GENERATED, self._on_code_generated)
        self.bus.subscribe(EventType.CODE_FIXED, self._on_code_fixed)
        self.bus.subscribe(EventType.EXECUTE_REQUESTED, self._on_execute_requested)

    async def _on_code_generated(self, event: Event):
        script_path = event.payload.get("script_path", "")
        strategy_id = event.payload.get("strategy_id", "")
        strategy = event.payload.get("strategy", {})
        if script_path:
            await self.execute_and_report(script_path, strategy_id, strategy, attempt=1)

    async def _on_code_fixed(self, event: Event):
        script_path = event.payload.get("script_path", "")
        strategy_id = event.payload.get("strategy_id", "")
        attempt = event.payload.get("attempt", 1)
        if script_path:
            await self.execute_and_report(script_path, strategy_id, {}, attempt=attempt)

    async def _on_execute_requested(self, event: Event):
        script_path = event.payload.get("script_path", "")
        strategy_id = event.payload.get("strategy_id", "")
        await self.execute_and_report(script_path, strategy_id, {})

    async def execute_and_report(
        self,
        script_path: str,
        strategy_id: str,
        strategy: dict,
        attempt: int = 1,
    ):
        """Execute script, parse output, route to evaluation or fix."""
        self.status = AgentStatus.WORKING
        self.logger.info(f"[Execution] Running: {script_path} (attempt {attempt})")
        await self.notify_telegram(
            f"⚡ **Executing Script** (attempt {attempt})\n"
            f"📄 `{script_path.split('/')[-1]}`",
            level="info"
        )

        try:
            result = await self.runner.run_script(script_path)

            store = get_store()
            await store.log_agent_action(
                self.agent_id, self.role, "execute_script",
                {
                    "script_path": script_path,
                    "strategy_id": strategy_id,
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "execution_time": result.execution_time,
                }
            )

            if result.success:
                # Parse JSON output
                metrics = {}
                try:
                    # Look for JSON in stdout
                    stdout = result.stdout.strip()
                    # Find last JSON object in output
                    import re
                    json_matches = re.findall(r'\{[^{}]*\}', stdout, re.DOTALL)
                    if json_matches:
                        for m in reversed(json_matches):
                            try:
                                metrics = json.loads(m)
                                break
                            except Exception:
                                continue
                except Exception:
                    metrics = {"raw_output": result.stdout[:500]}

                self.tasks_completed += 1
                self.status = AgentStatus.IDLE

                await self.notify_telegram(
                    f"✅ **Script Executed Successfully**\n"
                    f"⏱ Time: `{result.execution_time:.2f}s`\n"
                    f"📊 Metrics: `{str(metrics)[:150]}`",
                    level="success"
                )

                # Send to evaluation
                await self.emit(
                    EventType.BACKTEST_COMPLETED,
                    payload={
                        "strategy_id": strategy_id,
                        "script_path": script_path,
                        "metrics": metrics,
                        "stdout": result.stdout,
                        "execution_time": result.execution_time,
                        "strategy": strategy,
                    }
                )

            else:
                # Script failed — check if it's a missing module error first
                error_context = self.runner.extract_error_context(result)
                self.logger.warning(
                    f"[Execution] Script failed: {result.error_type} - {result.error_message}"
                )

                # ── Auto-install missing packages before asking CodingAgent ──
                if result.error_type in ("ModuleNotFoundError", "ImportError"):
                    installed_pkg = await self.pkg_manager.install_from_error(
                        result.stderr
                    )
                    if installed_pkg:
                        await self.notify_telegram(
                            f"📦 **Auto-installed package**: `{installed_pkg}`\n"
                            f"🔄 Retrying script automatically...",
                            level="info"
                        )
                        # Retry immediately with same script — package now available
                        await self.execute_and_report(
                            script_path, strategy_id, strategy, attempt=attempt
                        )
                        return

                # ── Also pre-scan the code for all imports and install proactively ──
                try:
                    with open(script_path, "r") as f:
                        code = f.read()
                    newly_installed = await self.pkg_manager.install_from_imports(code)
                    if newly_installed:
                        await self.notify_telegram(
                            f"📦 **Pre-installed packages**: `{', '.join(newly_installed)}`\n"
                            f"🔄 Retrying with all dependencies...",
                            level="info"
                        )
                        retry_result = await self.runner.run_script(script_path)
                        if retry_result.success:
                            # Success after package install — treat as success
                            result = retry_result
                            # Fall through to success handler below
                            import json, re as _re
                            metrics = {}
                            try:
                                stdout = retry_result.stdout.strip()
                                json_matches = _re.findall(r'\{[^{}]*\}', stdout, _re.DOTALL)
                                for m in reversed(json_matches):
                                    try:
                                        metrics = json.loads(m)
                                        break
                                    except Exception:
                                        continue
                            except Exception:
                                metrics = {}
                            self.tasks_completed += 1
                            self.status = AgentStatus.IDLE
                            await self.emit(
                                EventType.BACKTEST_COMPLETED,
                                payload={
                                    "strategy_id": strategy_id,
                                    "script_path": script_path,
                                    "metrics": metrics,
                                    "stdout": retry_result.stdout,
                                    "execution_time": retry_result.execution_time,
                                    "strategy": strategy,
                                }
                            )
                            return
                except Exception:
                    pass

                await self.notify_telegram(
                    f"⚠️ **Script Failed** (attempt {attempt}/3)\n"
                    f"🐛 Error: `{result.error_type}: {result.error_message}`\n"
                    f"🔧 Sending to Coding Agent for fix...",
                    level="warning"
                )

                if attempt <= 3:
                    await self.emit(
                        EventType.CODE_FIX_REQUESTED,
                        payload={
                            "strategy_id": strategy_id,
                            "script_path": script_path,
                            "error_context": error_context,
                            "attempt": attempt,
                        }
                    )
                else:
                    await self.emit(
                        EventType.BACKTEST_FAILED,
                        payload={
                            "strategy_id": strategy_id,
                            "script_path": script_path,
                            "error": error_context,
                        }
                    )

                self.tasks_failed += 1
                self.status = AgentStatus.IDLE

        except Exception as e:
            await self.handle_error(e, f"execute {script_path}")
            self.status = AgentStatus.IDLE
