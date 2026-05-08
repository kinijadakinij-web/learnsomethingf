"""
telegram_bot/bot.py — Luxury Telegram notification + control bot
Real-time agent updates with rich formatting and inline controls
"""
import asyncio
import io
import logging
import os
import time
import zipfile
from typing import Optional

from telegram import (
    Bot,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
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
    - 📦 Export Latest AI — sends the best strategy .py file on demand
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

        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start",   self._cmd_start))
        self._app.add_handler(CommandHandler("status",  self._cmd_status))
        self._app.add_handler(CommandHandler("top",     self._cmd_top_strategies))
        self._app.add_handler(CommandHandler("stats",   self._cmd_stats))
        self._app.add_handler(CommandHandler("pause",   self._cmd_pause))
        self._app.add_handler(CommandHandler("resume",  self._cmd_resume))
        self._app.add_handler(CommandHandler("export",  self._cmd_export))
        self._app.add_handler(CommandHandler("help",    self._cmd_help))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))

        bus = get_bus()
        bus.subscribe(EventType.TELEGRAM_NOTIFY, self._on_notify_event)

        asyncio.create_task(self._message_sender())
        asyncio.create_task(self._run_polling())

        await self._send_startup_message()
        logger.info("[Telegram] Bot started")

    async def stop(self):
        self._running = False
        if self._app:
            await self._app.stop()

    # ── Event handler ─────────────────────────────────────────────────────────

    async def _on_notify_event(self, event: Event):
        message = event.payload.get("message", "")
        level   = event.payload.get("level", "info")
        agent   = event.payload.get("agent", "")
        await self._enqueue_message(message, level, agent)

    async def _enqueue_message(self, text: str, level: str = "info", agent: str = ""):
        if self._message_queue:
            await self._message_queue.put((text, level, agent, time.time()))

    async def _message_sender(self):
        while self._running:
            try:
                text, level, agent, ts = await asyncio.wait_for(
                    self._message_queue.get(), timeout=1.0
                )
                await self._send_notification(text, level, agent)
                await asyncio.sleep(0.5)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[Telegram] Message sender error: {e}")

    async def _send_notification(self, text: str, level: str = "info", agent: str = ""):
        style = self.LEVEL_STYLES.get(level, self.LEVEL_STYLES["info"])
        timestamp = time.strftime("%H:%M:%S")
        msg = (
            f"{style['emoji']} {text}\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
            f"🕐 `{timestamp}`"
        )
        await self._send(msg)

    async def _send(self, text: str, reply_markup=None):
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

    # ── Startup ───────────────────────────────────────────────────────────────

    async def _send_startup_message(self):
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status",         callback_data="status"),
                InlineKeyboardButton("🏆 Top Strategies", callback_data="top"),
            ],
            [
                InlineKeyboardButton("📈 Stats",          callback_data="stats"),
                InlineKeyboardButton("⏸ Pause Lab",       callback_data="pause"),
            ],
            [
                InlineKeyboardButton("📦 Export Latest AI", callback_data="export_latest"),
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

    # ── Main keyboard (reusable) ──────────────────────────────────────────────

    def _main_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status",           callback_data="status"),
                InlineKeyboardButton("🏆 Top",              callback_data="top"),
            ],
            [
                InlineKeyboardButton("📈 Stats",            callback_data="stats"),
                InlineKeyboardButton("⏸ Pause",             callback_data="pause"),
            ],
            [
                InlineKeyboardButton("📦 Export Latest AI", callback_data="export_latest"),
            ],
        ])

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = f"""
🤖 *{config.LAB_NAME}* — Command Center

{DIVIDER}
📋 **COMMANDS**

/status  — Current agent status & activity
/top     — Top performing strategies
/stats   — Lab statistics
/pause   — Pause the research loop
/resume  — Resume the research loop
/export  — 📦 Export best strategy as .py file
/help    — This message

{DIVIDER}
💡 The lab runs autonomously.
Agents research → code → backtest → improve → repeat.
""".strip()
        await update.message.reply_text(
            msg,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self._main_keyboard(),
        )

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._cmd_help(update, ctx)

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_status(update)

    async def _cmd_top_strategies(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_top_strategies(update)

    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_stats(update)

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._lab_paused = True
        bus = get_bus()
        await bus.emit("system.pause", source_agent="telegram", payload={"paused": True})
        await update.message.reply_text(
            "⏸ **Lab paused.** Use /resume to restart.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._lab_paused = False
        bus = get_bus()
        await bus.emit("system.resume", source_agent="telegram", payload={"paused": False})
        await update.message.reply_text(
            "▶️ **Lab resumed.** Agents are back to work!",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_export(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Export best strategy .py file to Telegram."""
        await self._do_export(chat_id=update.message.chat_id)

    # ── Export logic ──────────────────────────────────────────────────────────

    async def _do_export(self, chat_id: int | str):
        """
        Find the best evaluated strategy that has a script file,
        then send it as a .py document to Telegram.
        If multiple versions exist (evolution chain), bundle them in a zip.
        """
        if not self._bot:
            return

        await self._bot.send_message(
            chat_id=chat_id,
            text="📦 *Preparing export...* Looking for best strategy.",
            parse_mode=ParseMode.MARKDOWN,
        )

        store = get_store()

        # 1. Get top strategies by score that have a script_path
        top = await store.get_best_strategies(limit=20)
        candidates = [s for s in top if s.get("script_path") and os.path.exists(s["script_path"])]

        if not candidates:
            # Fallback: latest strategy with a file, regardless of score
            all_strats = await store.get_all_strategies(limit=50)
            candidates = [s for s in all_strats if s.get("script_path") and os.path.exists(s["script_path"])]

        if not candidates:
            await self._bot.send_message(
                chat_id=chat_id,
                text=(
                    "❌ *No strategy files found yet.*\n\n"
                    "The lab hasn't finished generating and backtesting a strategy. "
                    "Try again in a few minutes."
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        best = candidates[0]
        strategy_name = best.get("strategy_name", "unknown")
        version       = best.get("version", 1)
        score         = best.get("score", 0)
        metrics       = best.get("metrics", {})
        script_path   = best["script_path"]
        strategy_id   = best.get("strategy_id", "")

        # 2. Collect entire evolution chain (all versions of this lineage)
        chain_files = await self._collect_evolution_chain(store, best)

        # 3. Build zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path, arcname in chain_files:
                try:
                    zf.write(file_path, arcname)
                except Exception:
                    pass

            # Also add a README summary
            readme = self._build_readme(best, chain_files)
            zf.writestr("README.md", readme)

        zip_buffer.seek(0)

        # 4. Build caption
        wr  = float(metrics.get("winrate", 0))
        pf  = float(metrics.get("profit_factor", 0))
        dd  = float(metrics.get("max_drawdown", metrics.get("max_drawdown_pct", 0)))
        caption = (
            f"📦 *Export: {strategy_name}*\n"
            f"{DIVIDER}\n"
            f"🏆 Score: `{score:.1f}` | v`{version}`\n"
            f"📈 WinRate: `{wr:.1%}` | PF: `{pf:.2f}`\n"
            f"📉 Max DD: `{dd:.1%}`\n"
            f"💱 Symbol: `{best.get('symbol','?')}` | TF: `{best.get('timeframe','?')}`\n"
            f"⚡ Leverage: `{best.get('leverage', 10)}x`\n"
            f"{DIVIDER}\n"
            f"📁 Contains `{len(chain_files)}` file(s) + README"
        )

        safe_name = strategy_name.lower().replace(" ", "_")[:30]
        zip_name  = f"strategy_{safe_name}_v{version}.zip"

        await self._bot.send_document(
            chat_id=chat_id,
            document=InputFile(zip_buffer, filename=zip_name),
            caption=caption,
            parse_mode=ParseMode.MARKDOWN,
        )

        logger.info(f"[Telegram] Exported strategy: {strategy_name} v{version}")

    async def _collect_evolution_chain(
        self, store, best: dict
    ) -> list[tuple[str, str]]:
        """
        Walk up the parent_id chain to collect all related strategy files.
        Returns list of (absolute_path, archive_name) tuples.
        """
        files = []
        seen_ids = set()

        # Start from the best strategy and walk its ancestors
        current = best
        while current:
            sid = current.get("strategy_id", "")
            if sid in seen_ids:
                break
            seen_ids.add(sid)

            sp = current.get("script_path", "")
            if sp and os.path.exists(sp):
                version = current.get("version", 1)
                arcname = f"v{version}_{os.path.basename(sp)}"
                files.append((sp, arcname))

            # Walk to parent
            parent_id = current.get("parent_id")
            if not parent_id:
                break

            all_strats = await store.get_all_strategies(limit=200)
            parent = next((s for s in all_strats if s.get("strategy_id") == parent_id), None)
            current = parent

        # Sort by version ascending
        files.sort(key=lambda x: x[1])
        return files

    @staticmethod
    def _build_readme(best: dict, chain_files: list) -> str:
        metrics = best.get("metrics", {})
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        lines = [
            f"# {best.get('strategy_name', 'Strategy Export')}",
            f"",
            f"**Exported:** {ts}",
            f"**Version:** {best.get('version', 1)}",
            f"**Symbol:** {best.get('symbol', 'BTCUSDT')}",
            f"**Timeframe:** {best.get('timeframe', '1h')}",
            f"**Leverage:** {best.get('leverage', 10)}x",
            f"",
            f"## Backtest Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Win Rate | {float(metrics.get('winrate', 0)):.1%} |",
            f"| Profit Factor | {float(metrics.get('profit_factor', 0)):.2f} |",
            f"| Net PnL | {float(metrics.get('pnl_pct', metrics.get('net_pnl_pct', 0))):.2f}% |",
            f"| Max Drawdown | {float(metrics.get('max_drawdown', metrics.get('max_drawdown_pct', 0))):.1%} |",
            f"| Total Trades | {int(metrics.get('total_trades', 0))} |",
            f"| Score | {float(best.get('score', 0)):.1f} |",
            f"",
            f"## Evolution Chain",
            f"",
        ]
        for _, arcname in chain_files:
            lines.append(f"- `{arcname}`")

        lines += [
            f"",
            f"## Strategy Logic",
            f"",
            f"**Entry Long:** {best.get('entry_long', best.get('entry_conditions', 'N/A'))}",
            f"",
            f"**Entry Short:** {best.get('entry_short', 'N/A')}",
            f"",
            f"**Exit Long:** {best.get('exit_long', best.get('exit_conditions', 'N/A'))}",
            f"",
            f"**Stop Loss:** {float(best.get('stop_loss_pct', 0.02)) * 100:.1f}%",
            f"",
            f"**Take Profit:** {float(best.get('take_profit_pct', 0.04)) * 100:.1f}%",
            f"",
            f"## Description",
            f"",
            f"{best.get('description', best.get('overview', 'No description.'))}",
        ]
        return "\n".join(lines)

    # ── Callback router ───────────────────────────────────────────────────────

    async def _on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        if data == "status":
            await self._send_status(update)
        elif data == "top":
            await self._send_top_strategies(update)
        elif data == "stats":
            await self._send_stats(update)
        elif data == "pause":
            self._lab_paused = True
            bus = get_bus()
            await bus.emit("system.pause", source_agent="telegram", payload={"paused": True})
            await query.message.reply_text(
                "⏸ **Lab paused.** Use /resume or the button above.",
                parse_mode=ParseMode.MARKDOWN,
            )
        elif data == "export_latest":
            await self._do_export(chat_id=query.message.chat_id)

    # ── Shared send helpers ───────────────────────────────────────────────────

    async def _send_stats(self, update):
        store = get_store()
        stats = await store.get_stats()
        uptime_secs = int(time.time() - self._start_time)
        uptime_str  = f"{uptime_secs // 3600}h {(uptime_secs % 3600) // 60}m"

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

        kb = self._main_keyboard()
        if hasattr(update, "message") and update.message:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
        else:
            await self._send(msg, reply_markup=kb)

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

        kb = self._main_keyboard()
        if hasattr(update, "message") and update.message:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
        else:
            await self._send(msg, reply_markup=kb)

    async def _send_top_strategies(self, update):
        store = get_store()
        top = await store.get_best_strategies(limit=5)

        if not top:
            msg = "🏆 *Top Strategies*\n\nNo strategies evaluated yet."
        else:
            lines  = []
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

        kb = self._main_keyboard()
        if hasattr(update, "message") and update.message:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
        else:
            await self._send(msg, reply_markup=kb)

    # ── Polling ───────────────────────────────────────────────────────────────

    async def _run_polling(self):
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

    # ── Public helpers ────────────────────────────────────────────────────────

    async def send_strategy_report(self, strategy: dict, metrics: dict):
        name   = strategy.get("strategy_name", "Unknown")
        wr     = float(metrics.get("winrate", 0))
        pf     = float(metrics.get("profit_factor", 0))
        dd     = float(metrics.get("max_drawdown_pct", 0))
        trades = int(metrics.get("total_trades", 0))
        pnl    = float(metrics.get("net_pnl_pct", 0))
        score  = float(metrics.get("score", 0))

        bar_len = 10
        wr_bar  = "█" * int(wr * bar_len) + "░" * (bar_len - int(wr * bar_len))
        grade   = (
            "🌟 EXCELLENT" if score > 60
            else "✅ GOOD"    if score > 40
            else "⚠️ MEDIOCRE" if score > 20
            else "❌ POOR"
        )

        msg = f"""
📊 *Strategy Report*

{DIVIDER}
🏷 **Name**: `{name}`
🎯 **Grade**: {grade}
{DIVIDER}
📈 **Performance**
• WinRate: `{wr:.1%}` `{wr_bar}`
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

        await self._send(msg, reply_markup=self._main_keyboard())


# Global singleton
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
