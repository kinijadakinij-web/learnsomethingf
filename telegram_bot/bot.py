"""
telegram_bot/bot.py — Luxury Telegram notification + control bot
Real-time agent updates with rich formatting and inline controls
"""
import asyncio
import logging
import time
from typing import Optional

from telegram import (
    Bot,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

import config
from core.event_bus import EventType, Event, get_bus
from memory.mongodb_store import get_store

logger = logging.getLogger(__name__)

# ── Luxury formatting helpers ──────────────────────────────────────────────────

HEADER = """
╔══════════════════════════════╗
║   🤖  QUANT LAB  ALPHA       ║
╚══════════════════════════════╝
""".strip()

DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


def fmt_metric(label: str, value: str, emoji: str = "•") -> str:
    return f"{emoji} `{label}`: **{value}**"


def fmt_section(title: str, lines: list[str]) -> str:
    body = "\n".join(f"  {l}" for l in lines)
    return f"**{title}**\n{body}"


class TelegramNotifier:
    """
    Luxury Telegram bot for the Trading Lab.
    
    Features:
    - Real-time agent activity notifications
    - Strategy performance reports
    - Inline controls (pause/resume lab, request status)
    - Formatted, beautiful messages
    - Level-based styling (info/success/warning/error)
    """

    LEVEL_STYLES = {
        "info":    {"prefix": "ℹ️",  "emoji": "💬"},
        "success": {"prefix": "✅",  "emoji": "🌟"},
        "warning": {"prefix": "⚠️",  "emoji": "🟡"},
        "error":   {"prefix": "❌",  "emoji": "🔴"},
        "system":  {"prefix": "⚙️",  "emoji": "🔧"},
    }

    def __init__(self):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self._bot: Optional[Bot] = None
        self._app: Optional[Application] = None
        self._running = False
        self._message_queue: asyncio.Queue = None
        self._lab_paused = False
        self._start_time = time.time()

    async def start(self):
        if not self.token or not self.chat_id:
            logger.warning("[Telegram] Bot token or chat_id not configured — skipping")
            return

        self._message_queue = asyncio.Queue()
        self._bot = Bot(token=self.token)
        self._running = True

        # Setup command handlers
        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("top", self._cmd_top_strategies))
        self._app.add_handler(CommandHandler("stats", self._cmd_stats))
        self._app.add_handler(CommandHandler("pause", self._cmd_pause))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))

        # Subscribe to event bus notifications
        bus = get_bus()
        bus.subscribe(EventType.TELEGRAM_NOTIFY, self._on_notify_event)

        # Start message sender coroutine
        asyncio.create_task(self._message_sender())

        # Start bot polling
        asyncio.create_task(self._run_polling())

        await self._send_startup_message()
        logger.info("[Telegram] Bot started")

    async def stop(self):
        self._running = False
        if self._app:
            await self._app.stop()

    # ─── Event Handler ────────────────────────────────────────────────────────

    async def _on_notify_event(self, event: Event):
        message = event.payload.get("message", "")
        level = event.payload.get("level", "info")
        agent = event.payload.get("agent", "")
        await self._enqueue_message(message, level, agent)

    async def _enqueue_message(self, text: str, level: str = "info", agent: str = ""):
        if self._message_queue:
            await self._message_queue.put((text, level, agent, time.time()))

    async def _message_sender(self):
        """Drains the message queue and sends to Telegram."""
        while self._running:
            try:
                text, level, agent, ts = await asyncio.wait_for(
                    self._message_queue.get(), timeout=1.0
                )
                await self._send_notification(text, level, agent)
                await asyncio.sleep(0.5)  # Rate limiting
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[Telegram] Message sender error: {e}")

    async def _send_notification(self, text: str, level: str = "info", agent: str = ""):
        """Format and send a notification message."""
        style = self.LEVEL_STYLES.get(level, self.LEVEL_STYLES["info"])
        timestamp = time.strftime("%H:%M:%S")

        msg = (
            f"{style['emoji']} {text}\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
            f"🕐 `{timestamp}`"
        )

        await self._send(msg)

    async def _send(self, text: str, reply_markup=None):
        """Raw send to Telegram."""
        if not self._bot or not self.chat_id:
            return
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning(f"[Telegram] Send failed: {e}")

    # ─── Startup Message ──────────────────────────────────────────────────────

    async def _send_startup_message(self):
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("🏆 Top Strategies", callback_data="top"),
            ],
            [
                InlineKeyboardButton("📈 Stats", callback_data="stats"),
                InlineKeyboardButton("⏸ Pause Lab", callback_data="pause"),
            ],
        ])

        uptime_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

        msg = f"""
🚀 *{config.LAB_NAME}* is now **LIVE**

{DIVIDER}
🤖 **Autonomous AI Trading Research Lab**
🔬 Multi-agent evolution system initialized
💾 MongoDB connected
🧠 Qwen AI pool: `{len(config.QWEN_BEARERS)} tokens`
{DIVIDER}

📡 *Monitoring active* — I'll send updates as agents work.

Use /help to see all commands.
`{uptime_str}`
""".strip()

        await self._send(msg, reply_markup=keyboard)

    # ─── Commands ─────────────────────────────────────────────────────────────

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = f"""
🤖 *{config.LAB_NAME}* — Command Center

{DIVIDER}
📋 **COMMANDS**

/status — Current agent status & activity
/top — Top performing strategies
/stats — Lab statistics (strategies, backtests)
/pause — Pause the research loop
/resume — Resume the research loop
/help — This message

{DIVIDER}
💡 The lab runs autonomously.
Agents research → code → backtest → improve → repeat.
""".strip()
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._cmd_help(update, ctx)

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_status(update)

    async def _cmd_top_strategies(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_top_strategies(update)

    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        store = get_store()
        stats = await store.get_stats()
        uptime_secs = int(time.time() - self._start_time)
        uptime_str = f"{uptime_secs // 3600}h {(uptime_secs % 3600) // 60}m"

        msg = f"""
📈 *Lab Statistics*

{DIVIDER}
🔬 Strategies created: `{stats.get('total_strategies', 0)}`
📊 Backtests run: `{stats.get('total_backtests', 0)}`
🧪 Research reports: `{stats.get('total_research', 0)}`
🧬 Evolution nodes: `{stats.get('total_evolution_nodes', 0)}`
⚡ Improvements: `{stats.get('total_improvements', 0)}`
⏱ Lab uptime: `{uptime_str}`
{DIVIDER}
""".strip()
        if update:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await self._send(msg)

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._lab_paused = True
        bus = get_bus()
        # Emit pause signal
        await bus.emit(
            "system.pause",
            source_agent="telegram",
            payload={"paused": True}
        )
        await update.message.reply_text(
            "⏸ **Lab paused.** Use /resume to restart.",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._lab_paused = False
        bus = get_bus()
        await bus.emit(
            "system.resume",
            source_agent="telegram",
            payload={"paused": False}
        )
        await update.message.reply_text(
            "▶️ **Lab resumed.** Agents are back to work!",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        if data == "status":
            await self._send_status(update)
        elif data == "top":
            await self._send_top_strategies(update)
        elif data == "stats":
            await self._cmd_stats(update, ctx)
        elif data == "pause":
            self._lab_paused = True
            await query.message.reply_text("⏸ Lab paused.", parse_mode=ParseMode.MARKDOWN)

    async def _send_status(self, update):
        bus = get_bus()
        recent = bus.recent_events(n=5)

        recent_lines = []
        for e in reversed(recent):
            ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            recent_lines.append(f"`{ts}` `{e.event_type.split('.')[-1]}` ← {e.source_agent[:8]}")

        msg = f"""
⚙️ *System Status*

{DIVIDER}
🟢 **Status**: {'PAUSED ⏸' if self._lab_paused else 'RUNNING ▶️'}
🧠 **AI Pool**: `{len(config.QWEN_BEARERS)} tokens`
📡 **Event Bus**: Active

{DIVIDER}
📋 **Recent Events:**
{chr(10).join(recent_lines) if recent_lines else 'None yet'}
{DIVIDER}
""".strip()

        if hasattr(update, "message") and update.message:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await self._send(msg)

    async def _send_top_strategies(self, update):
        store = get_store()
        top = await store.get_best_strategies(limit=5)

        if not top:
            msg = "🏆 *Top Strategies*\n\nNo strategies evaluated yet."
        else:
            lines = []
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for i, s in enumerate(top):
                m = s.get("metrics", {})
                lines.append(
                    f"{medals[i]} `{s.get('strategy_name', 'N/A')[:25]}`\n"
                    f"   Score: `{s.get('score', 0):.1f}` | "
                    f"WR: `{float(m.get('winrate', 0)):.1%}` | "
                    f"PF: `{float(m.get('profit_factor', 0)):.2f}`"
                )

            msg = f"""
🏆 *Top Performing Strategies*

{DIVIDER}
{chr(10).join(lines)}
{DIVIDER}
""".strip()

        if hasattr(update, "message") and update.message:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await self._send(msg)

    async def _run_polling(self):
        """Run the bot polling loop."""
        try:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
            while self._running:
                await asyncio.sleep(1)
            await self._app.updater.stop()
            await self._app.stop()
        except Exception as e:
            logger.error(f"[Telegram] Polling error: {e}")

    # ─── Public helpers ───────────────────────────────────────────────────────

    async def send_strategy_report(self, strategy: dict, metrics: dict):
        """Send a full formatted strategy report."""
        name = strategy.get("strategy_name", "Unknown")
        winrate = float(metrics.get("winrate", 0))
        pf = float(metrics.get("profit_factor", 0))
        dd = float(metrics.get("max_drawdown_pct", 0))
        trades = int(metrics.get("total_trades", 0))
        pnl = float(metrics.get("net_pnl_pct", 0))
        score = float(metrics.get("score", 0))

        bar_len = 10
        wr_bar = "█" * int(winrate * bar_len) + "░" * (bar_len - int(winrate * bar_len))

        grade = "🌟 EXCELLENT" if score > 60 else "✅ GOOD" if score > 40 else "⚠️ MEDIOCRE" if score > 20 else "❌ POOR"

        msg = f"""
📊 *Strategy Report*

{DIVIDER}
🏷 **Name**: `{name}`
🎯 **Grade**: {grade}
{DIVIDER}
📈 **Performance**
• WinRate: `{winrate:.1%}` `{wr_bar}`
• Profit Factor: `{pf:.2f}`
• Net PnL: `{pnl:+.1f}%`
• Max Drawdown: `{dd:.1%}`
• Total Trades: `{trades}`
• Score: `{score:.1f}/100`
{DIVIDER}
⚡ Leverage: `{strategy.get('leverage', 10)}x`
📅 Timeframe: `{strategy.get('timeframe', '?')}`
💱 Symbol: `{strategy.get('symbol', '?')}`
{DIVIDER}
""".strip()

        await self._send(msg)


# Global singleton
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
