from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client.http import models

from src.config import Settings, load_settings
from src.llm_engine import LlamaEngine
from src.qdrant_setup import QdrantStore, create_store


@dataclass
class IndexedMessage:
    chat_id: int
    chat_name: str
    message_id: int
    sender_id: int
    author_name: str
    timestamp: int
    text: str
    is_reply: bool
    reply_to_msg_id: int | None = None


@dataclass
class Burst:
    chat_id: int
    chat_name: str
    sender_id: int
    author_name: str
    timestamp: int
    message_ids: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    has_reply: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.texts).strip()

    @property
    def first_message_id(self) -> int:
        return min(self.message_ids) if self.message_ids else 0

    @property
    def last_message_id(self) -> int:
        return max(self.message_ids) if self.message_ids else 0


def normalize_author(username: str | None, user_id: int) -> str:
    if username:
        return f"@{username.lstrip('@')}"
    return f"@tg_{user_id}"


def should_index_message(is_private: bool, sender_id: int | None, me_id: int, replied_to_sender_id: int | None) -> bool:
    if is_private:
        return True
    if sender_id == me_id:
        return True
    if replied_to_sender_id == me_id:
        return True
    return False


def merge_messages_into_bursts(messages: list[IndexedMessage]) -> list[Burst]:
    if not messages:
        return []
    sorted_msgs = sorted(messages, key=lambda x: (x.timestamp, x.message_id))
    bursts: list[Burst] = []
    current: Burst | None = None

    for msg in sorted_msgs:
        if current is None:
            current = Burst(
                chat_id=msg.chat_id,
                chat_name=msg.chat_name,
                sender_id=msg.sender_id,
                author_name=msg.author_name,
                timestamp=msg.timestamp,
                message_ids=[msg.message_id],
                texts=[msg.text],
                has_reply=msg.is_reply,
            )
            continue

        time_diff = msg.timestamp - current.timestamp
        can_merge = (
            msg.sender_id == current.sender_id
            and time_diff < 60
            and not msg.is_reply
            and not current.has_reply
        )
        if can_merge:
            current.message_ids.append(msg.message_id)
            current.texts.append(msg.text)
            current.timestamp = msg.timestamp
            continue

        bursts.append(current)
        current = Burst(
            chat_id=msg.chat_id,
            chat_name=msg.chat_name,
            sender_id=msg.sender_id,
            author_name=msg.author_name,
            timestamp=msg.timestamp,
            message_ids=[msg.message_id],
            texts=[msg.text],
            has_reply=msg.is_reply,
        )

    if current is not None:
        bursts.append(current)
    return bursts


class IngestState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = {"chat_last_ids": {}}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self._state = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")

    def get_last_id(self, chat_id: int) -> int:
        return int(self._state.get("chat_last_ids", {}).get(str(chat_id), 0))

    def set_last_id(self, chat_id: int, message_id: int) -> None:
        self._state.setdefault("chat_last_ids", {})[str(chat_id)] = int(message_id)


def point_id_for_burst(chat_id: int, message_ids: list[int]) -> str:
    raw = f"{chat_id}:{','.join(str(m) for m in sorted(message_ids))}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TelegramIngestor:
    def __init__(self, settings: Settings, store: QdrantStore, llm: LlamaEngine, state: IngestState):
        self.settings = settings
        self.store = store
        self.llm = llm
        self.state = state

    async def run(self, mode: str) -> None:
        if mode not in {"full", "sync", "full_then_sync"}:
            raise ValueError("mode must be one of: full, sync, full_then_sync")

        if mode == "full":
            await self._run_full()
            return
        if mode == "sync":
            await self._run_sync()
            return
        await self._run_full()
        await self._run_sync()

    async def _run_full(self) -> None:
        client, me_id = await self._connect()
        try:
            async for dialog in client.iter_dialogs():
                await self._ingest_dialog(client, dialog, me_id, min_id=0)
            self.state.save()
        finally:
            await client.disconnect()

    async def _run_sync(self) -> None:
        client, me_id = await self._connect()
        try:
            async for dialog in client.iter_dialogs():
                min_id = self.state.get_last_id(dialog.id)
                await self._ingest_dialog(client, dialog, me_id, min_id=min_id)
            self.state.save()
            await self._run_live(client, me_id)
        finally:
            await client.disconnect()

    async def _run_live(self, client: Any, me_id: int) -> None:
        from telethon import events

        @client.on(events.NewMessage())
        async def handler(event: Any) -> None:
            indexed = await self._extract_indexed_messages(
                client=client,
                dialog_name=getattr(event.chat, "title", None) or "private",
                chat_id=event.chat_id,
                is_private=bool(event.is_private),
                message=event.message,
                me_id=me_id,
            )
            if not indexed:
                return
            await self._index_messages(indexed)
            self.state.set_last_id(int(event.chat_id), int(event.message.id))
            self.state.save()

        await client.run_until_disconnected()

    async def _connect(self) -> tuple[Any, int]:
        from telethon import TelegramClient

        if not self.settings.telegram_api_id or not self.settings.telegram_api_hash:
            raise RuntimeError("Missing TELEGRAM_API_ID/TELEGRAM_API_HASH in .env")

        client = TelegramClient(
            self.settings.telegram_session_name,
            self.settings.telegram_api_id,
            self.settings.telegram_api_hash,
        )
        await client.start(phone=self.settings.telegram_phone)
        me = await client.get_me()
        return client, int(me.id)

    async def _ingest_dialog(self, client: Any, dialog: Any, me_id: int, min_id: int) -> None:
        chat_id = int(dialog.id)
        chat_name = getattr(dialog, "name", None) or getattr(dialog.entity, "title", None) or "unknown_chat"
        is_private = bool(getattr(dialog, "is_user", False))

        indexed_messages: list[IndexedMessage] = []
        max_seen_message_id = min_id

        async for message in client.iter_messages(dialog.entity, reverse=True, min_id=min_id):
            if not getattr(message, "message", None):
                continue
            max_seen_message_id = max(max_seen_message_id, int(message.id))
            extracted = await self._extract_indexed_messages(
                client=client,
                dialog_name=chat_name,
                chat_id=chat_id,
                is_private=is_private,
                message=message,
                me_id=me_id,
            )
            indexed_messages.extend(extracted)

        await self._index_messages(indexed_messages)
        if max_seen_message_id > 0:
            self.state.set_last_id(chat_id, max_seen_message_id)

    async def _extract_indexed_messages(
        self,
        client: Any,
        dialog_name: str,
        chat_id: int,
        is_private: bool,
        message: Any,
        me_id: int,
    ) -> list[IndexedMessage]:
        sender_id = int(getattr(message, "sender_id", 0) or 0)
        sender = await message.get_sender()
        username = getattr(sender, "username", None) if sender else None
        author = normalize_author(username=username, user_id=sender_id)
        replied_to_sender_id = None
        original_reply_message = None

        if getattr(message, "reply_to_msg_id", None):
            original_reply_message = await message.get_reply_message()
            if original_reply_message is not None:
                replied_to_sender_id = int(getattr(original_reply_message, "sender_id", 0) or 0)

        should_index = should_index_message(
            is_private=is_private,
            sender_id=sender_id,
            me_id=me_id,
            replied_to_sender_id=replied_to_sender_id,
        )
        if not should_index:
            return []

        created = int(message.date.replace(tzinfo=timezone.utc).timestamp()) if message.date else int(datetime.now(timezone.utc).timestamp())
        items = [
            IndexedMessage(
                chat_id=chat_id,
                chat_name=dialog_name,
                message_id=int(message.id),
                sender_id=sender_id,
                author_name=author,
                timestamp=created,
                text=str(message.message).strip(),
                is_reply=bool(getattr(message, "reply_to_msg_id", None)),
                reply_to_msg_id=int(message.reply_to_msg_id) if getattr(message, "reply_to_msg_id", None) else None,
            )
        ]

        if sender_id == me_id and original_reply_message is not None and getattr(original_reply_message, "message", None):
            orig_sender = await original_reply_message.get_sender()
            orig_sender_id = int(getattr(original_reply_message, "sender_id", 0) or 0)
            orig_username = getattr(orig_sender, "username", None) if orig_sender else None
            orig_author = normalize_author(username=orig_username, user_id=orig_sender_id)
            orig_created = (
                int(original_reply_message.date.replace(tzinfo=timezone.utc).timestamp())
                if original_reply_message.date
                else created
            )
            items.append(
                IndexedMessage(
                    chat_id=chat_id,
                    chat_name=dialog_name,
                    message_id=int(original_reply_message.id),
                    sender_id=orig_sender_id,
                    author_name=orig_author,
                    timestamp=orig_created,
                    text=str(original_reply_message.message).strip(),
                    is_reply=bool(getattr(original_reply_message, "reply_to_msg_id", None)),
                    reply_to_msg_id=int(original_reply_message.reply_to_msg_id)
                    if getattr(original_reply_message, "reply_to_msg_id", None)
                    else None,
                )
            )
        return [item for item in items if item.text]

    async def _index_messages(self, indexed_messages: list[IndexedMessage]) -> None:
        if not indexed_messages:
            return
        bursts = merge_messages_into_bursts(indexed_messages)
        points: list[models.PointStruct] = []
        for burst in bursts:
            if not burst.text:
                continue
            vector = self.llm.embed_text(burst.text)
            payload = {
                "text": burst.text,
                "author_name": burst.author_name,
                "chat_name": burst.chat_name,
                "timestamp": burst.timestamp,
                "message_ids": burst.message_ids,
                "chat_id": burst.chat_id,
                "first_message_id": burst.first_message_id,
                "last_message_id": burst.last_message_id,
            }
            points.append(
                models.PointStruct(
                    id=point_id_for_burst(burst.chat_id, burst.message_ids),
                    vector=vector,
                    payload=payload,
                )
            )
        self.store.upsert_points(points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Telegram messages into Qdrant")
    parser.add_argument(
        "--mode",
        default="sync",
        choices=["full", "sync", "full_then_sync"],
        help="Ingestion mode",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    settings = load_settings()
    store = create_store(settings)
    llm = LlamaEngine(settings)
    state = IngestState(settings.ingest_state_path)
    ingestor = TelegramIngestor(settings=settings, store=store, llm=llm, state=state)
    await ingestor.run(args.mode)


if __name__ == "__main__":
    asyncio.run(_main())
