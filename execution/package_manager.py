"""
execution/package_manager.py — Dynamic Package Manager
Allows AI agents to install any Python package on-the-fly.
No requirements.txt limitation — agents decide what they need.
"""
import asyncio
import logging
import re
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Packages that are NEVER allowed (security)
BLACKLIST = {
    "os-sys", "setup-tools", "py-req", "acquiressl",
    "colourama",  # known malicious typosquats
}

# Cache of already-installed packages this session (avoid re-installing)
_installed_cache: set[str] = set()
_failed_cache: set[str] = set()


class PackageManager:
    """
    Dynamic pip installer for AI-generated scripts.

    Features:
    - Auto-detect missing imports from ModuleNotFoundError
    - Install any package on demand
    - Cache installed packages (no duplicate installs)
    - Security blacklist for known malicious packages
    - Supports package aliases (e.g. "sklearn" → "scikit-learn")
    - Updates requirements.txt automatically
    """

    # Maps import name → pip package name (when they differ)
    IMPORT_TO_PACKAGE = {
        "sklearn":       "scikit-learn",
        "cv2":           "opencv-python-headless",
        "PIL":           "Pillow",
        "bs4":           "beautifulsoup4",
        "yaml":          "PyYAML",
        "dotenv":        "python-dotenv",
        "ta":            "ta",
        "talib":         "TA-Lib",
        "pandas_ta":     "pandas-ta",
        "finta":         "finta",
        "vectorbt":      "vectorbt",
        "backtrader":    "backtrader",
        "zipline":       "zipline-reloaded",
        "pyfolio":       "pyfolio-reloaded",
        "empyrical":     "empyrical",
        "statsmodels":   "statsmodels",
        "lightgbm":      "lightgbm",
        "xgboost":       "xgboost",
        "catboost":      "catboost",
        "torch":         "torch",
        "tensorflow":    "tensorflow",
        "keras":         "keras",
        "stable_baselines3": "stable-baselines3",
        "gym":           "gymnasium",
        "gymnasium":     "gymnasium",
        "optuna":        "optuna",
        "plotly":        "plotly",
        "mplfinance":    "mplfinance",
        "ccxt":          "ccxt",
        "websocket":     "websocket-client",
        "pymongo":       "pymongo",
        "motor":         "motor",
        "redis":         "redis",
        "scipy":         "scipy",
        "numba":         "numba",
        "cython":        "Cython",
        "arch":          "arch",
        "pykalman":      "pykalman",
        "hmmlearn":      "hmmlearn",
        "ruptures":      "ruptures",
        "stumpy":        "stumpy",
        "tsfresh":       "tsfresh",
        "prophet":       "prophet",
        "neuralprophet": "neuralprophet",
    }

    def __init__(self):
        self.python = sys.executable
        self._lock = asyncio.Lock()

    async def install(
        self,
        package: str,
        version: Optional[str] = None,
        upgrade: bool = False,
    ) -> bool:
        """
        Install a package via pip.
        Returns True if installed successfully (or already installed).
        """
        # Normalize package name
        pip_name = self.IMPORT_TO_PACKAGE.get(package, package)
        cache_key = pip_name.lower().split("==")[0].split(">=")[0]

        if cache_key in _installed_cache:
            logger.debug(f"[PackageManager] '{pip_name}' already installed (cached)")
            return True

        if cache_key in _failed_cache:
            logger.warning(f"[PackageManager] '{pip_name}' previously failed, skipping")
            return False

        if cache_key in BLACKLIST:
            logger.error(f"[PackageManager] '{pip_name}' is BLACKLISTED — refusing install")
            return False

        async with self._lock:
            install_spec = f"{pip_name}=={version}" if version else pip_name
            cmd = [self.python, "-m", "pip", "install", install_spec, "-q"]
            if upgrade:
                cmd.append("--upgrade")

            logger.info(f"[PackageManager] Installing: {install_spec}")
            start = time.time()

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=120
                )
                elapsed = time.time() - start

                if proc.returncode == 0:
                    _installed_cache.add(cache_key)
                    logger.info(
                        f"[PackageManager] ✓ Installed '{pip_name}' in {elapsed:.1f}s"
                    )
                    # Update requirements.txt
                    await self._append_to_requirements(pip_name)
                    return True
                else:
                    _failed_cache.add(cache_key)
                    err = stderr.decode("utf-8", errors="replace")[:300]
                    logger.warning(
                        f"[PackageManager] ✗ Failed to install '{pip_name}': {err}"
                    )
                    return False

            except asyncio.TimeoutError:
                logger.warning(f"[PackageManager] Install timeout for '{pip_name}'")
                _failed_cache.add(cache_key)
                return False
            except Exception as e:
                logger.error(f"[PackageManager] Install error for '{pip_name}': {e}")
                _failed_cache.add(cache_key)
                return False

    async def install_from_imports(self, code: str) -> list[str]:
        """
        Parse Python code, extract all imports, install any missing packages.
        Returns list of packages installed.
        """
        imports = self._extract_imports(code)
        installed = []

        for imp in imports:
            if await self._is_installed(imp):
                continue
            # Try to install
            success = await self.install(imp)
            if success:
                installed.append(imp)

        return installed

    async def install_from_error(self, error_message: str) -> Optional[str]:
        """
        Parse a ModuleNotFoundError and install the missing package.
        Returns the package name if installed successfully.
        """
        # Extract module name from error
        # "No module named 'xyz'" or "ModuleNotFoundError: No module named 'xyz.abc'"
        match = re.search(
            r"No module named ['\"]([^'\"]+)['\"]", error_message
        )
        if not match:
            return None

        module_name = match.group(1).split(".")[0]  # top-level package only
        logger.info(f"[PackageManager] Auto-installing missing module: '{module_name}'")

        success = await self.install(module_name)
        return module_name if success else None

    async def install_many(self, packages: list[str]) -> dict[str, bool]:
        """Install multiple packages. Returns {package: success} dict."""
        results = {}
        for pkg in packages:
            results[pkg] = await self.install(pkg)
        return results

    @staticmethod
    async def _is_installed(import_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", f"import {import_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)
            return proc.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _extract_imports(code: str) -> list[str]:
        """Extract top-level package names from Python code."""
        packages = set()
        patterns = [
            r"^import\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import",
        ]
        for line in code.split("\n"):
            line = line.strip()
            for pattern in patterns:
                m = re.match(pattern, line)
                if m:
                    pkg = m.group(1)
                    # Skip stdlib and already-known safe ones
                    if pkg not in _STDLIB_MODULES:
                        packages.add(pkg)
        return list(packages)

    @staticmethod
    async def _append_to_requirements(package: str):
        """Add a newly installed package to requirements.txt."""
        import os
        import config
        req_path = os.path.join(config.BASE_DIR, "requirements.txt")
        try:
            with open(req_path, "r") as f:
                existing = f.read()
            # Check not already listed
            pkg_base = package.lower().split("==")[0].split(">=")[0]
            if pkg_base not in existing.lower():
                with open(req_path, "a") as f:
                    f.write(f"\n{package}")
                logger.debug(f"[PackageManager] Added '{package}' to requirements.txt")
        except Exception:
            pass  # Non-critical


# Standard library modules — never try to pip install these
_STDLIB_MODULES = {
    "os", "sys", "re", "json", "time", "math", "random", "uuid", "abc",
    "ast", "io", "csv", "datetime", "pathlib", "shutil", "glob", "fnmatch",
    "subprocess", "threading", "multiprocessing", "asyncio", "concurrent",
    "socket", "ssl", "http", "urllib", "email", "html", "xml", "sqlite3",
    "logging", "unittest", "traceback", "inspect", "types", "typing",
    "functools", "itertools", "collections", "heapq", "bisect", "array",
    "struct", "hashlib", "hmac", "base64", "binascii", "copy", "pickle",
    "contextlib", "weakref", "gc", "platform", "signal", "stat", "tempfile",
    "warnings", "dataclasses", "enum", "string", "textwrap", "pprint",
    "decimal", "fractions", "statistics", "operator", "builtins",
}


# Global singleton
_pm: Optional[PackageManager] = None


def get_package_manager() -> PackageManager:
    global _pm
    if _pm is None:
        _pm = PackageManager()
    return _pm
