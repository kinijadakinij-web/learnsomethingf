"""
execution/runner.py — Safe, sandboxed Python script execution engine
Runs generated scripts with timeout, crash recovery, output capture
"""
import asyncio
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    script_path: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ScriptRunner:
    """
    Safely executes Python scripts in isolated subprocess.
    Features:
    - Timeout protection
    - Output capture (stdout + stderr)
    - Error classification
    - Retry on transient failures
    - Execution logging
    """

    def __init__(
        self,
        timeout: int = None,
        max_retries: int = 1,
        python_path: str = None,
    ):
        self.timeout = timeout or config.EXECUTION_TIMEOUT
        self.max_retries = max_retries
        self.python_path = python_path or sys.executable

    async def run_script(
        self,
        script_path: str,
        args: Optional[list] = None,
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a Python script and return results."""
        cmd = [self.python_path, script_path] + (args or [])
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=run_env,
                    cwd=cwd or config.GENERATED_DIR,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    elapsed = time.time() - start_time
                    logger.warning(
                        f"[Runner] Script timed out after {elapsed:.1f}s: {script_path}"
                    )
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=f"Execution timed out after {self.timeout}s",
                        exit_code=-1,
                        execution_time=elapsed,
                        script_path=script_path,
                        error_type="TimeoutError",
                        error_message=f"Script exceeded {self.timeout}s timeout",
                    )

                elapsed = time.time() - start_time
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = proc.returncode

                success = exit_code == 0

                error_type = None
                error_message = None

                if not success and stderr:
                    error_type, error_message = self._classify_error(stderr)

                result = ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    execution_time=elapsed,
                    script_path=script_path,
                    error_type=error_type,
                    error_message=error_message,
                )

                log_msg = (
                    f"[Runner] {'✓' if success else '✗'} "
                    f"{os.path.basename(script_path)} "
                    f"({elapsed:.2f}s, exit={exit_code})"
                )
                if success:
                    logger.info(log_msg)
                else:
                    logger.warning(f"{log_msg}\n  Error: {error_message}")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[Runner] Exception on attempt {attempt+1}: {e}")
                if attempt == self.max_retries - 1:
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=traceback.format_exc(),
                        exit_code=-1,
                        execution_time=elapsed,
                        script_path=script_path,
                        error_type="RuntimeError",
                        error_message=str(e),
                    )
                await asyncio.sleep(1)

    async def run_code_string(
        self,
        code: str,
        filename: str = "temp_script.py",
        args: Optional[list] = None,
    ) -> ExecutionResult:
        """Write code to a temp file and execute it."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="lab_",
            dir=config.GENERATED_DIR,
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            return await self.run_script(temp_path, args=args)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    @staticmethod
    def _classify_error(stderr: str) -> tuple[str, str]:
        """Classify Python errors for targeted fixing."""
        error_patterns = {
            "SyntaxError": "SyntaxError",
            "ImportError": "ImportError",
            "ModuleNotFoundError": "ModuleNotFoundError",
            "NameError": "NameError",
            "TypeError": "TypeError",
            "ValueError": "ValueError",
            "IndexError": "IndexError",
            "KeyError": "KeyError",
            "AttributeError": "AttributeError",
            "ZeroDivisionError": "ZeroDivisionError",
            "FileNotFoundError": "FileNotFoundError",
            "MemoryError": "MemoryError",
        }

        for error_name, error_type in error_patterns.items():
            if error_name in stderr:
                # Extract the error message line
                lines = stderr.strip().split("\n")
                error_lines = [l for l in lines if error_name in l]
                message = error_lines[-1] if error_lines else lines[-1]
                return error_type, message.strip()

        # Generic error
        lines = stderr.strip().split("\n")
        return "UnknownError", lines[-1] if lines else "Unknown error"

    def extract_error_context(self, result: ExecutionResult) -> str:
        """Format error context for the AI to fix."""
        if result.success:
            return ""

        context = f"""
EXECUTION ERROR
===============
Script: {result.script_path}
Exit Code: {result.exit_code}
Error Type: {result.error_type}
Error Message: {result.error_message}

STDERR (last 50 lines):
{chr(10).join(result.stderr.split(chr(10))[-50:])}

STDOUT (last 20 lines):
{chr(10).join(result.stdout.split(chr(10))[-20:])}
""".strip()
        return context
