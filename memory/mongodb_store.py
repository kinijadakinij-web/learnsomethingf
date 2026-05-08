"""
memory/mongodb_store.py — MongoDB persistence layer using motor (async)
Handles all long-term memory for strategies, results, agent history
"""
import logging
import time
from typing import Any, Dict, List, Optional

import motor.motor_asyncio
from bson import ObjectId

import config

logger = logging.getLogger(__name__)


class MongoStore:
    """
    Async MongoDB store using motor.
    Central repository for all Trading Lab data.
    """

    # Collections
    COL_STRATEGIES    = "strategies"
    COL_BACKTESTS     = "backtests"
    COL_RESEARCH      = "research"
    COL_AGENT_LOGS    = "agent_logs"
    COL_EVOLUTION     = "evolution_tree"
    COL_MEMORY        = "memory_bank"
    COL_IMPROVEMENTS  = "improvements"

    def __init__(self):
        self._client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self._db = None

    async def connect(self):
        self._client = motor.motor_asyncio.AsyncIOMotorClient(config.MONGODB_URI)
        self._db = self._client[config.MONGODB_DB]
        # Ensure indexes
        await self._setup_indexes()
        logger.info(f"[MongoDB] Connected to {config.MONGODB_DB}")

    async def disconnect(self):
        if self._client:
            self._client.close()
            logger.info("[MongoDB] Disconnected")

    async def _setup_indexes(self):
        """Create indexes for performance."""
        await self._db[self.COL_STRATEGIES].create_index("strategy_id", unique=True)
        await self._db[self.COL_STRATEGIES].create_index("created_at")
        await self._db[self.COL_STRATEGIES].create_index("score")
        await self._db[self.COL_BACKTESTS].create_index("strategy_id")
        await self._db[self.COL_BACKTESTS].create_index("created_at")
        await self._db[self.COL_RESEARCH].create_index("created_at")
        await self._db[self.COL_EVOLUTION].create_index("parent_id")

    # ─── Strategy CRUD ────────────────────────────────────────────────────────

    async def save_strategy(self, strategy: dict) -> str:
        """Save or update a strategy. Returns strategy_id."""
        strategy["updated_at"] = time.time()
        if "created_at" not in strategy:
            strategy["created_at"] = time.time()

        sid = strategy.get("strategy_id")
        if sid:
            await self._db[self.COL_STRATEGIES].update_one(
                {"strategy_id": sid},
                {"$set": strategy},
                upsert=True,
            )
        else:
            import uuid
            strategy["strategy_id"] = str(uuid.uuid4())
            await self._db[self.COL_STRATEGIES].insert_one(strategy)

        return strategy["strategy_id"]

    async def get_strategy(self, strategy_id: str) -> Optional[dict]:
        doc = await self._db[self.COL_STRATEGIES].find_one({"strategy_id": strategy_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def get_best_strategies(self, limit: int = 10) -> List[dict]:
        cursor = self._db[self.COL_STRATEGIES].find(
            {"score": {"$exists": True}},
            sort=[("score", -1)],
            limit=limit,
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    async def get_all_strategies(self, limit: int = 50) -> List[dict]:
        cursor = self._db[self.COL_STRATEGIES].find(
            {}, sort=[("created_at", -1)], limit=limit
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ─── Backtest CRUD ────────────────────────────────────────────────────────

    async def save_backtest(self, result: dict) -> str:
        result["created_at"] = time.time()
        res = await self._db[self.COL_BACKTESTS].insert_one(result)
        return str(res.inserted_id)

    async def get_backtests_for_strategy(
        self, strategy_id: str, limit: int = 10
    ) -> List[dict]:
        cursor = self._db[self.COL_BACKTESTS].find(
            {"strategy_id": strategy_id},
            sort=[("created_at", -1)],
            limit=limit,
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ─── Research CRUD ────────────────────────────────────────────────────────

    async def save_research(self, research: dict) -> str:
        research["created_at"] = time.time()
        res = await self._db[self.COL_RESEARCH].insert_one(research)
        return str(res.inserted_id)

    async def get_recent_research(self, limit: int = 20) -> List[dict]:
        cursor = self._db[self.COL_RESEARCH].find(
            {}, sort=[("created_at", -1)], limit=limit
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ─── Agent Logs ───────────────────────────────────────────────────────────

    async def log_agent_action(self, agent_id: str, role: str, action: str, data: dict):
        await self._db[self.COL_AGENT_LOGS].insert_one({
            "agent_id": agent_id,
            "role": role,
            "action": action,
            "data": data,
            "timestamp": time.time(),
        })

    async def get_agent_logs(self, role: str = None, limit: int = 50) -> List[dict]:
        query = {"role": role} if role else {}
        cursor = self._db[self.COL_AGENT_LOGS].find(
            query, sort=[("timestamp", -1)], limit=limit
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ─── Evolution Tree ───────────────────────────────────────────────────────

    async def record_evolution(self, parent_id: str, child_id: str, reason: str):
        await self._db[self.COL_EVOLUTION].insert_one({
            "parent_id": parent_id,
            "child_id": child_id,
            "reason": reason,
            "timestamp": time.time(),
        })

    async def get_evolution_tree(self, strategy_id: str) -> List[dict]:
        """Get full evolution chain for a strategy."""
        results = []
        current_id = strategy_id
        for _ in range(50):  # max depth 50
            doc = await self._db[self.COL_EVOLUTION].find_one(
                {"child_id": current_id}
            )
            if not doc:
                break
            doc["_id"] = str(doc["_id"])
            results.append(doc)
            current_id = doc["parent_id"]
        return list(reversed(results))

    # ─── Memory Bank ─────────────────────────────────────────────────────────

    async def remember(self, key: str, value: Any, category: str = "general"):
        """Store a key-value memory."""
        await self._db[self.COL_MEMORY].update_one(
            {"key": key, "category": category},
            {"$set": {"value": value, "updated_at": time.time()}},
            upsert=True,
        )

    async def recall(self, key: str, category: str = "general") -> Optional[Any]:
        """Retrieve a stored memory."""
        doc = await self._db[self.COL_MEMORY].find_one(
            {"key": key, "category": category}
        )
        return doc["value"] if doc else None

    async def recall_category(self, category: str) -> List[dict]:
        """Get all memories in a category."""
        cursor = self._db[self.COL_MEMORY].find({"category": category})
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ─── Stats ────────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        return {
            "total_strategies": await self._db[self.COL_STRATEGIES].count_documents({}),
            "total_backtests": await self._db[self.COL_BACKTESTS].count_documents({}),
            "total_research": await self._db[self.COL_RESEARCH].count_documents({}),
            "total_improvements": await self._db[self.COL_IMPROVEMENTS].count_documents({}),
            "total_evolution_nodes": await self._db[self.COL_EVOLUTION].count_documents({}),
        }


# Global singleton
_store: Optional[MongoStore] = None


def get_store() -> MongoStore:
    global _store
    if _store is None:
        _store = MongoStore()
    return _store
