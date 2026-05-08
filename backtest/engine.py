"""
backtest/engine.py — Futures backtest simulation engine
Supports leverage, fees, liquidation, slippage, SL/TP, partial close
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str                    # "long" | "short"
    entry_price: float
    exit_price: Optional[float]
    size: float                  # contracts
    leverage: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fee: float = 0.0
    liquidated: bool = False
    exit_reason: str = ""        # "sl" | "tp" | "signal" | "liquidation"


@dataclass
class BacktestResult:
    strategy_id: str
    symbol: str
    timeframe: str
    initial_capital: float
    final_capital: float
    net_pnl: float
    net_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    winrate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float
    expectancy: float
    total_fees: int
    liquidations: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    score: float = 0.0           # composite score

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("trades", "equity_curve")}
        d["equity_curve_len"] = len(self.equity_curve)
        return d


class BacktestEngine:
    """
    Futures backtest engine with:
    - Long/short support
    - Leverage and liquidation
    - Maker/taker fees
    - Slippage simulation
    - Stop-loss and take-profit
    - Performance metrics
    """

    def __init__(
        self,
        initial_capital: float = None,
        leverage: int = None,
        fee_rate: float = None,
        slippage: float = None,
        liquidation_threshold: float = 0.80,
    ):
        self.initial_capital = initial_capital or config.DEFAULT_INITIAL_CAPITAL
        self.leverage = leverage or config.DEFAULT_LEVERAGE
        self.fee_rate = fee_rate or config.DEFAULT_FEE_RATE
        self.slippage = slippage or config.DEFAULT_SLIPPAGE
        self.liquidation_threshold = liquidation_threshold

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        stop_loss_pct: Optional[float] = 0.02,
        take_profit_pct: Optional[float] = 0.04,
        strategy_id: str = "unknown",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> BacktestResult:
        """
        Run backtest on OHLCV dataframe with signals.

        df: DataFrame with columns [open, high, low, close, volume]
        signals: Series with values 1 (long), -1 (short), 0 (flat)
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        capital = self.initial_capital
        equity_curve = [capital]
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        peak_capital = capital

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_signal = signals.iloc[i - 1]
            current_signal = signals.iloc[i]
            price = row["close"]

            # ── Check liquidation on open trades ──────────────────────────────
            if current_trade is not None:
                liq_price = self._liquidation_price(current_trade)
                if current_trade.side == "long" and row["low"] <= liq_price:
                    pnl, fee = self._close_trade(current_trade, liq_price, capital)
                    capital += pnl - fee
                    current_trade.pnl = pnl
                    current_trade.fee = fee
                    current_trade.exit_price = liq_price
                    current_trade.exit_time = row.name
                    current_trade.liquidated = True
                    current_trade.exit_reason = "liquidation"
                    trades.append(current_trade)
                    current_trade = None
                    equity_curve.append(max(0, capital))
                    continue

                elif current_trade.side == "short" and row["high"] >= liq_price:
                    pnl, fee = self._close_trade(current_trade, liq_price, capital)
                    capital += pnl - fee
                    current_trade.pnl = pnl
                    current_trade.fee = fee
                    current_trade.exit_price = liq_price
                    current_trade.exit_time = row.name
                    current_trade.liquidated = True
                    current_trade.exit_reason = "liquidation"
                    trades.append(current_trade)
                    current_trade = None
                    equity_curve.append(max(0, capital))
                    continue

            # ── Check SL/TP on open trades ────────────────────────────────────
            if current_trade is not None and stop_loss_pct and take_profit_pct:
                sl_hit, tp_hit, exit_price, reason = self._check_sl_tp(
                    current_trade, row, stop_loss_pct, take_profit_pct
                )
                if sl_hit or tp_hit:
                    pnl, fee = self._close_trade(current_trade, exit_price, capital)
                    capital += pnl - fee
                    current_trade.pnl = pnl
                    current_trade.fee = fee
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = row.name
                    current_trade.exit_reason = reason
                    trades.append(current_trade)
                    current_trade = None

            # ── Signal-based entries/exits ────────────────────────────────────
            if prev_signal != current_signal:
                # Close existing trade
                if current_trade is not None:
                    exit_p = price * (1 + self.slippage)
                    pnl, fee = self._close_trade(current_trade, exit_p, capital)
                    capital += pnl - fee
                    current_trade.pnl = pnl
                    current_trade.fee = fee
                    current_trade.exit_price = exit_p
                    current_trade.exit_time = row.name
                    current_trade.exit_reason = "signal"
                    trades.append(current_trade)
                    current_trade = None

                # Open new trade
                if current_signal in (1, -1) and capital > 0:
                    side = "long" if current_signal == 1 else "short"
                    entry_p = price * (1 + self.slippage if side == "long" else 1 - self.slippage)
                    size = (capital * self.leverage) / entry_p
                    current_trade = Trade(
                        entry_time=row.name,
                        exit_time=None,
                        side=side,
                        entry_price=entry_p,
                        exit_price=None,
                        size=size,
                        leverage=self.leverage,
                    )

            equity_curve.append(capital)
            peak_capital = max(peak_capital, capital)

        # Close any open trade at end
        if current_trade is not None:
            last_price = df.iloc[-1]["close"]
            pnl, fee = self._close_trade(current_trade, last_price, capital)
            capital += pnl - fee
            current_trade.pnl = pnl
            current_trade.fee = fee
            current_trade.exit_price = last_price
            current_trade.exit_time = df.index[-1]
            current_trade.exit_reason = "end"
            trades.append(current_trade)

        return self._compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_capital=capital,
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _liquidation_price(self, trade: Trade) -> float:
        margin_rate = 1 / trade.leverage
        if trade.side == "long":
            return trade.entry_price * (1 - margin_rate * self.liquidation_threshold)
        else:
            return trade.entry_price * (1 + margin_rate * self.liquidation_threshold)

    def _close_trade(
        self, trade: Trade, exit_price: float, capital: float
    ) -> Tuple[float, float]:
        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size
        fee = (trade.entry_price + exit_price) * trade.size * self.fee_rate
        return pnl, fee

    def _check_sl_tp(
        self,
        trade: Trade,
        row,
        sl_pct: float,
        tp_pct: float,
    ) -> Tuple[bool, bool, float, str]:
        if trade.side == "long":
            sl_price = trade.entry_price * (1 - sl_pct)
            tp_price = trade.entry_price * (1 + tp_pct)
            if row["low"] <= sl_price:
                return True, False, sl_price, "sl"
            if row["high"] >= tp_price:
                return False, True, tp_price, "tp"
        else:
            sl_price = trade.entry_price * (1 + sl_pct)
            tp_price = trade.entry_price * (1 - tp_pct)
            if row["high"] >= sl_price:
                return True, False, sl_price, "sl"
            if row["low"] <= tp_price:
                return False, True, tp_price, "tp"
        return False, False, 0.0, ""

    def _compute_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        initial_capital: float,
        final_capital: float,
        strategy_id: str,
        symbol: str,
        timeframe: str,
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=initial_capital,
                final_capital=final_capital,
                net_pnl=final_capital - initial_capital,
                net_pnl_pct=(final_capital - initial_capital) / initial_capital * 100,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                winrate=0,
                profit_factor=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                avg_win=0,
                avg_loss=0,
                risk_reward_ratio=0,
                expectancy=0,
                total_fees=0,
                liquidations=0,
                equity_curve=equity_curve,
            )

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_fees = sum(t.fee for t in trades)
        liquidations = sum(1 for t in trades if t.liquidated)

        winrate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0.001
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
        expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)

        # Max drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_dd_pct = float(np.max(drawdown)) if len(drawdown) > 0 else 0
        max_dd = float(np.max(peak - eq)) if len(drawdown) > 0 else 0

        # Sharpe ratio (simplified, daily returns approximation)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-10)
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)
        else:
            sharpe = 0

        net_pnl = final_capital - initial_capital
        net_pnl_pct = (net_pnl / initial_capital) * 100

        # Composite score (higher = better strategy)
        score = (
            (winrate * 30)
            + (min(profit_factor, 5) / 5 * 25)
            + (max(0, net_pnl_pct) / 100 * 20)
            + (max(0, sharpe) / 3 * 15)
            - (max_dd_pct * 10)
        )

        result = BacktestResult(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_capital,
            final_capital=final_capital,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            winrate=winrate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_reward_ratio=risk_reward,
            expectancy=expectancy,
            total_fees=int(total_fees),
            liquidations=liquidations,
            trades=trades,
            equity_curve=equity_curve,
            score=score,
        )

        logger.info(
            f"[Backtest] {strategy_id[:8]} | "
            f"Trades: {len(trades)} | WR: {winrate:.1%} | "
            f"PF: {profit_factor:.2f} | DD: {max_dd_pct:.1%} | Score: {score:.1f}"
        )

        return result
