"""
config.py — Central configuration for Trading Lab
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Qwen ──────────────────────────────────────────────────────────────────────
QWEN_BEARERS: list[str] = [
    t.strip() for t in os.getenv("QWEN_BEARERS", "").split(",") if t.strip()
]
QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen3-max")
QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://chat.qwen.ai")

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "trading_lab")

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── MEXC ──────────────────────────────────────────────────────────────────────
MEXC_API_KEY: str = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY: str = os.getenv("MEXC_SECRET_KEY", "")
MEXC_BASE_URL: str = "https://contract.mexc.com"
MEXC_WS_URL: str = "wss://contract.mexc.com/edge"

# ── System ────────────────────────────────────────────────────────────────────
LAB_NAME: str = os.getenv("LAB_NAME", "QuantLab-Alpha")
MAX_CONCURRENT_AGENTS: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
LOOP_INTERVAL_SECONDS: int = int(os.getenv("LOOP_INTERVAL_SECONDS", "300"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
EXECUTION_TIMEOUT: int = int(os.getenv("EXECUTION_TIMEOUT", "120"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR: str = os.path.join(BASE_DIR, "generated")
STRATEGIES_DIR: str = os.path.join(GENERATED_DIR, "strategies")
REPORTS_DIR: str = os.path.join(GENERATED_DIR, "reports")
BACKTESTS_DIR: str = os.path.join(GENERATED_DIR, "backtests")
LOGS_DIR: str = os.path.join(GENERATED_DIR, "logs")

# ── Backtest Defaults ─────────────────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL: float = 10_000.0
DEFAULT_LEVERAGE: int = 10
DEFAULT_FEE_RATE: float = 0.0005   # 0.05% taker
DEFAULT_SLIPPAGE: float = 0.0002   # 0.02%

# ── Evolution ─────────────────────────────────────────────────────────────────
MIN_WINRATE_THRESHOLD: float = 0.50
MIN_PROFIT_FACTOR: float = 1.3
MAX_DRAWDOWN_THRESHOLD: float = 0.25
