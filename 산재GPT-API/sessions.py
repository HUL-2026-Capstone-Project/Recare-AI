"""
대화 세션 저장소.

REDIS_URL이 있으면 Redis, 없으면 프로세스 메모리에 저장한다.
메모리 저장소는 uvicorn을 --workers 2 이상으로 띄우면 워커끼리 세션이 공유되지 않아 멀티턴이 끊긴다.
운영 환경에서는 Redis를 쓴다.
"""
import json
import os

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))  # 마지막 대화 후 7일
MAX_STORED_TURNS = 50  # 세션당 보관하는 최대 대화 수 (답변 생성에는 최근 몇 턴만 쓴다)

Turn = tuple[str, str]  # (사용자 질문, AI 답변)


class MemorySessionStore:
    def __init__(self):
        self._data: dict[str, list[Turn]] = {}

    async def get(self, session_id: str) -> list[Turn]:
        return list(self._data.get(session_id, []))

    async def append(self, session_id: str, question: str, answer: str) -> int:
        turns = self._data.setdefault(session_id, [])
        turns.append((question, answer))
        del turns[:-MAX_STORED_TURNS]
        return len(turns)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._data

    async def delete(self, session_id: str) -> bool:
        return self._data.pop(session_id, None) is not None

    async def ping(self) -> str:
        return "memory"


class RedisSessionStore:
    """세션 하나를 Redis 리스트 하나(session:{id})에 [질문, 답변] JSON으로 쌓는다."""

    def __init__(self, url: str):
        import redis.asyncio as redis  # REDIS_URL을 쓸 때만 필요한 의존성

        self._redis = redis.from_url(url, decode_responses=True)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    async def get(self, session_id: str) -> list[Turn]:
        return [tuple(json.loads(x)) for x in await self._redis.lrange(self._key(session_id), 0, -1)]

    async def append(self, session_id: str, question: str, answer: str) -> int:
        key = self._key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps([question, answer], ensure_ascii=False))
            pipe.ltrim(key, -MAX_STORED_TURNS, -1)
            pipe.expire(key, SESSION_TTL_SECONDS)  # 대화할 때마다 만료 시간을 연장
            pipe.llen(key)
            *_, length = await pipe.execute()
        return length

    async def exists(self, session_id: str) -> bool:
        return bool(await self._redis.exists(self._key(session_id)))

    async def delete(self, session_id: str) -> bool:
        return bool(await self._redis.delete(self._key(session_id)))

    async def ping(self) -> str:
        await self._redis.ping()
        return "redis"


def create_session_store():
    url = os.getenv("REDIS_URL")
    return RedisSessionStore(url) if url else MemorySessionStore()
