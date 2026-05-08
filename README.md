# 🤖 QuantLab Alpha — Autonomous AI Futures Trading Research Lab

> A self-evolving multi-agent AI system that autonomously researches, builds, backtests, evaluates, and improves crypto futures trading strategies — endlessly.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│          (Manages loop, monitors agents, routes)         │
└───────────────────┬─────────────────────────────────────┘
                    │ Event Bus (async pub/sub)
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌────────┐    ┌──────────┐    ┌──────────┐
│Research│    │ Strategy │    │  Memory  │
│ Agent  │───▶│  Agent   │    │  Agent   │
└────────┘    └────┬─────┘    └──────────┘
                   │
                   ▼
             ┌──────────┐
             │  Coding  │◀──────────┐
             │  Agent   │           │ (fix loop)
             └────┬─────┘           │
                  │                 │
                  ▼                 │
          ┌────────────┐    ┌───────┴────┐
          │ Execution  │───▶│   Coding   │
          │   Agent    │    │(Fix Agent) │
          └─────┬──────┘    └────────────┘
                │
                ▼
        ┌───────────────┐
        │  Evaluation   │
        │    Agent      │
        └──────┬────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐  ┌────────────┐
│Improvement │  │   Risk     │
│   Agent    │  │   Agent    │
└─────┬──────┘  └────────────┘
      │
      ▼ (generates new strategy → loops back to CodingAgent)
```

## 🔁 Evolution Loop

```
Research ──▶ Strategy ──▶ Code ──▶ Execute ──▶ Evaluate ──▶ Improve ──┐
    ▲                              (fix if error)         │            │
    └──────────────────────────────────────────────────── ▼            │
                              [next cycle]          [child strategy] ──┘
```

---

## 📁 Project Structure

```
trading_lab/
├── main.py                    # Entry point
├── config.py                  # All config from .env
├── requirements.txt
├── Procfile                   # Railway
├── railway.toml
│
├── ai/
│   ├── qwen_client.py         # Qwen reverse API client
│   └── bearer_pool.py         # Multi-token rotation pool
│
├── core/
│   ├── agent_base.py          # Base class for all agents
│   ├── event_bus.py           # Async pub/sub event system
│   └── orchestrator.py        # Main lab controller
│
├── agents/
│   ├── orchestrator_agent.py  # Strategic decision maker
│   ├── research_agent.py      # Generates research ideas
│   ├── strategy_agent.py      # Converts ideas to specs
│   ├── coding_agent.py        # Writes Python scripts
│   ├── execution_agent.py     # Runs scripts + error recovery
│   ├── evaluation_agent.py    # Grades strategy performance
│   ├── improvement_agent.py   # Evolves strategies
│   ├── memory_agent.py        # Tracks history + patterns
│   └── risk_agent.py          # Evaluates risk metrics
│
├── backtest/
│   └── engine.py              # Full futures backtest engine
│
├── market_data/
│   ├── mexc_client.py         # MEXC REST + WebSocket
│   └── synthetic_data.py      # Synthetic OHLCV generator
│
├── memory/
│   └── mongodb_store.py       # MongoDB async persistence
│
├── execution/
│   └── runner.py              # Safe subprocess executor
│
├── file_manager/
│   └── manager.py             # File I/O helper
│
├── telegram_bot/
│   └── bot.py                 # Luxury Telegram bot
│
└── generated/
    ├── strategies/            # AI-generated Python scripts
    ├── reports/               # Evaluation reports
    ├── backtests/             # Backtest JSON results
    └── logs/                  # Execution logs
```

---

## ⚙️ Setup

### 1. Clone & install
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`
```bash
cp .env.example .env
# Edit .env with your values
```

Required:
```env
QWEN_BEARERS=token1,token2,token3
MONGODB_URI=mongodb://...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### 3. Run locally
```bash
python main.py
```

### 4. Deploy to Railway
```bash
railway up
```

Set environment variables in Railway dashboard from `.env.example`.

---

## 🤖 Qwen Bearer Tokens

Get from `chat.qwen.ai` → browser DevTools → Network → any request → `Authorization: Bearer <token>`

Add multiple tokens (comma-separated) for higher throughput and automatic rotation:
```
QWEN_BEARERS=eyJ...,eyJ...,eyJ...
```

The `BearerPool` auto-rotates, tracks failures, and applies cooldowns on bad tokens.

---

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Current agent status |
| `/top` | Top performing strategies |
| `/stats` | Lab statistics |
| `/pause` | Pause research loop |
| `/resume` | Resume research loop |
| `/help` | Command list |

---

## 🗄 MongoDB Collections

| Collection | Contents |
|------------|----------|
| `strategies` | All strategies (spec + metrics + evaluation) |
| `backtests` | Raw backtest results |
| `research` | Research reports |
| `evolution_tree` | Parent→child strategy relationships |
| `agent_logs` | Per-agent action history |
| `memory_bank` | Key-value institutional memory |

---

## 🔧 Key Design Decisions

- **Qwen Reverse API**: No API cost — uses browser session tokens
- **Multi-Bearer Pool**: Auto-rotation prevents rate limits
- **Event Bus**: Loose coupling between agents — easy to add new ones
- **MongoDB + Motor**: Fully async, Railway-compatible
- **Synthetic Data**: Scripts run without any external API
- **Error Recovery Loop**: CodingAgent fixes broken scripts automatically (up to 3 attempts)
- **Evolution Tree**: Full lineage tracked for every strategy

---

## 📈 Roadmap

- [ ] Real MEXC data integration
- [ ] WebSocket live monitoring
- [ ] Reinforcement Learning agent
- [ ] Multi-symbol parallel backtesting
- [ ] Strategy ensemble / portfolio builder
- [ ] Fine-tuning pipeline on best strategies
- [ ] Paper trading with live signals
- [ ] Performance dashboard (web UI)
