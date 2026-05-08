"""
market_data/mexc_client.py — MEXC Futures Market Data Client
Abstraction layer for REST API + WebSocket streaming.
The AI agents will decide which mode to use based on their task.
"""
import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Callable, Dict, List, Optional

import aiohttp
import pandas as pd

import config

logger = logging.getLogger(__name__)


class MEXCRestClient:
    """
    MEXC Futures REST API client.
    Handles candle downloads, market info, ticker data.
    No auth required for public market data.
    """

    BASE = config.MEXC_BASE_URL

    TIMEFRAME_MAP = {
        "1m":  "Min1",
        "5m":  "Min5",
        "15m": "Min15",
        "30m": "Min30",
        "1h":  "Min60",
        "4h":  "Hour4",
        "8h":  "Hour8",
        "1d":  "Day1",
        "1w":  "Week1",
    }

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_klines(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Download historical OHLCV candles.
        Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        tf = self.TIMEFRAME_MAP.get(timeframe, "Min60")
        url = f"{self.BASE}/api/v1/contract/kline/{symbol}"
        params = {"interval": tf, "limit": limit}

        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            candles = data.get("data", {})
            if not candles:
                logger.warning(f"[MEXC] No data for {symbol} {timeframe}")
                return pd.DataFrame()

            # MEXC returns lists for each field
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(candles.get("time", []), unit="s"),
                "open":   [float(x) for x in candles.get("open", [])],
                "high":   [float(x) for x in candles.get("high", [])],
                "low":    [float(x) for x in candles.get("low", [])],
                "close":  [float(x) for x in candles.get("close", [])],
                "volume": [float(x) for x in candles.get("vol", [])],
            })
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"[MEXC] Downloaded {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[MEXC] get_klines error: {e}")
            return pd.DataFrame()

    async def get_ticker(self, symbol: str) -> dict:
        """Get latest ticker for a symbol."""
        url = f"{self.BASE}/api/v1/contract/ticker"
        session = await self._get_session()
        try:
            async with session.get(url, params={"symbol": symbol}) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data.get("data", {})
        except Exception as e:
            logger.error(f"[MEXC] get_ticker error: {e}")
            return {}

    async def get_all_symbols(self) -> List[str]:
        """Get list of all tradeable perpetual symbols."""
        url = f"{self.BASE}/api/v1/contract/detail"
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
            symbols = [
                item["symbol"]
                for item in data.get("data", [])
                if item.get("quoteCoin") == "USDT"
            ]
            return symbols
        except Exception as e:
            logger.error(f"[MEXC] get_all_symbols error: {e}")
            return ["BTC_USDT", "ETH_USDT"]

    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate."""
        url = f"{self.BASE}/api/v1/contract/funding_rate/{symbol}"
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return float(data.get("data", {}).get("fundingRate", 0))
        except Exception as e:
            logger.error(f"[MEXC] get_funding_rate error: {e}")
            return 0.0


class MEXCWebSocketClient:
    """
    MEXC Futures WebSocket client for real-time market data.
    Use for live monitoring when REST polling is insufficient.
    The AI agents decide when to switch to WebSocket mode.
    """

    WS_URL = config.MEXC_WS_URL

    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._ws = None
        self._subscribed = set()

    def on(self, channel: str, callback: Callable):
        """Subscribe a callback to a WebSocket channel."""
        if channel not in self._callbacks:
            self._callbacks[channel] = []
        self._callbacks[channel].append(callback)

    async def connect(self):
        """Connect and start receiving messages."""
        self._running = True
        asyncio.create_task(self._run())

    async def _run(self):
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.WS_URL) as ws:
                        self._ws = ws
                        logger.info("[MEXC WS] Connected")

                        # Re-subscribe to all channels
                        for channel in self._subscribed:
                            await self._subscribe_raw(channel)

                        # Start heartbeat
                        asyncio.create_task(self._heartbeat())

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(json.loads(msg.data))
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except Exception as e:
                logger.warning(f"[MEXC WS] Disconnected: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def subscribe_klines(self, symbol: str, timeframe: str = "Min1"):
        channel = f"sub.kline.{symbol}.{timeframe}"
        self._subscribed.add(channel)
        if self._ws:
            await self._subscribe_raw(channel)

    async def subscribe_ticker(self, symbol: str):
        channel = f"sub.ticker.{symbol}"
        self._subscribed.add(channel)
        if self._ws:
            await self._subscribe_raw(channel)

    async def _subscribe_raw(self, channel: str):
        await self._ws.send_json({"method": channel})

    async def _handle_message(self, data: dict):
        channel = data.get("channel", "")
        for registered, cbs in self._callbacks.items():
            if registered in channel:
                for cb in cbs:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(data)
                        else:
                            cb(data)
                    except Exception as e:
                        logger.error(f"[MEXC WS] Callback error: {e}")

    async def _heartbeat(self):
        while self._running and self._ws:
            try:
                await self._ws.send_json({"method": "ping"})
                await asyncio.sleep(20)
            except Exception:
                break

    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()


class MarketDataRouter:
    """
    Smart router — the AI agents call this to get data.
    It decides internally whether to use REST or WebSocket
    based on the request pattern and update frequency needed.
    """

    def __init__(self):
        self.rest = MEXCRestClient()
        self.ws = MEXCWebSocketClient()
        self._ws_active = False

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Always use REST for historical data."""
        return await self.rest.get_klines(symbol, timeframe, limit)

    async def start_live_feed(
        self,
        symbol: str,
        timeframe: str,
        on_candle: Callable,
    ):
        """Start live feed via WebSocket for real-time updates."""
        if not self._ws_active:
            await self.ws.connect()
            self._ws_active = True

        self.ws.on(f"kline.{symbol}", on_candle)
        await self.ws.subscribe_klines(symbol, timeframe)

    async def get_market_context(self, symbol: str) -> dict:
        """Get full market context for AI decision making."""
        ticker = await self.rest.get_ticker(symbol)
        funding = await self.rest.get_funding_rate(symbol)
        return {
            "symbol": symbol,
            "last_price": ticker.get("lastPrice", 0),
            "volume_24h": ticker.get("volume24", 0),
            "change_24h_pct": ticker.get("riseFallRate", 0),
            "funding_rate": funding,
            "timestamp": time.time(),
        }

    async def close(self):
        await self.rest.close()
        await self.ws.disconnect()
