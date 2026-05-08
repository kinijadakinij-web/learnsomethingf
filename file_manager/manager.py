"""
file_manager/manager.py — File management for the lab
Creates, reads, writes, organizes generated files
"""
import logging
import os
import time

import config

logger = logging.getLogger(__name__)


class FileManager:
    """Manages all file I/O for the trading lab."""

    def __init__(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        dirs = [
            config.GENERATED_DIR,
            config.STRATEGIES_DIR,
            config.REPORTS_DIR,
            config.BACKTESTS_DIR,
            config.LOGS_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def save_strategy_script(self, filename: str, code: str) -> str:
        path = os.path.join(config.STRATEGIES_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.debug(f"[FileManager] Saved strategy script: {path}")
        return path

    def save_report(self, filename: str, content: str) -> str:
        path = os.path.join(config.REPORTS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def save_file(self, path: str, content: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def list_strategies(self) -> list[str]:
        return [
            os.path.join(config.STRATEGIES_DIR, f)
            for f in os.listdir(config.STRATEGIES_DIR)
            if f.endswith(".py")
        ]

    def save_backtest_report(self, strategy_id: str, result: dict) -> str:
        import json
        filename = f"backtest_{strategy_id[:8]}_{int(time.time())}.json"
        path = os.path.join(config.BACKTESTS_DIR, filename)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        return path
