"""Utilities for interacting with database containing user conversations and preferences"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


class DatabaseAlreadyOpen(Exception):
    pass


class DatabaseNotOpen(Exception):
    pass


class UserDoesNotExist(Exception):
    def __init__(self, user_id: int, *args: object) -> None:
        self.user_id = user_id
        super().__init__(*args)


class CouldNotCreateUser(Exception):
    def __init__(self, user_id: int, *args: object) -> None:
        self.user_id = user_id
        super().__init__(*args)


@dataclass
class User:
    id: int
    system_prompt: str


@dataclass
class Message:
    id: int
    user_id: int
    timestamp: datetime
    position: int
    role: str
    message: str


def _requires_open_db(func) -> Callable:  # type: ignore
    def wrapper(self, *args, **kwargs) -> Callable:  # type: ignore
        if not self.is_open:
            raise DatabaseNotOpen()
        return func(self, *args, **kwargs)  # type: ignore

    return wrapper


class BotDatabase:
    def __init__(self, database_path: Path | str | None = None, default_system_prompt: str = "") -> None:
        self.default_system_prompt = default_system_prompt
        self.is_open = False
        if database_path is not None:
            self.open(database_path)

    def open(self, database_path: Path | str) -> None:
        if self.is_open:
            raise DatabaseAlreadyOpen()

        self.db = sqlite3.connect(database_path)
        self.is_open = True
        self._initialize_database()
        self.change_global_default_system_prompt(self.default_system_prompt)

    @_requires_open_db
    def close(self) -> None:
        self.db.close()
        self.is_open = False

    @_requires_open_db
    def change_global_default_system_prompt(self, new_system_prompt: str) -> int:
        with self.db as db:
            query = db.execute(
                "UPDATE users SET system_prompt = ? WHERE system_prompt == ?",
                (new_system_prompt, self.default_system_prompt),
            )
        self.default_system_prompt = new_system_prompt
        return query.rowcount

    @_requires_open_db
    def get_user(self, user_id: int) -> User | None:
        """Returns user or None if it doesn't exist."""
        query = self.db.execute("SELECT system_prompt FROM users WHERE id == ?", (user_id,))
        if system_prompt := query.fetchone():
            return User(user_id, system_prompt[0])
        return None

    @_requires_open_db
    def add_user(self, user_id: int, system_prompt: str | None = None) -> bool:
        """Returns True if adding was successful, False if user already exists.
        If system prompt is not provided, default one will be used."""
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        with self.db as db:
            query = db.execute(
                "INSERT INTO users(id, system_prompt) VALUES (?, ?)",
                (user_id, system_prompt),
            )
        return query.rowcount == 1

    @_requires_open_db
    def delete_user(self, user_id: int) -> bool:
        with self.db as db:
            query = db.execute("DELETE FROM users WHERE id == ?", (user_id,))
        return query.rowcount == 1

    @_requires_open_db
    def user_exists(self, user_id: int) -> bool:
        query = self.db.execute("SELECT EXISTS(SELECT 1 FROM users WHERE id == ?)", (user_id,))
        return bool(query.fetchone()[0] == 1)

    @_requires_open_db
    def change_user_system_prompt(
        self,
        user_id: int,
        new_system_prompt: str,
        create_user_if_not_found: bool = True,
    ) -> None:
        """Raises exception on user-related error."""
        if not self.user_exists(user_id):
            if create_user_if_not_found:
                self.add_user(user_id, new_system_prompt)
                return
            else:
                raise UserDoesNotExist(user_id)

        with self.db as db:
            db.execute(
                "UPDATE users SET system_prompt = ? WHERE id == ?",
                (new_system_prompt, user_id),
            )

    @_requires_open_db
    def get_message(self, message_id: int) -> Message | None:
        query = self.db.execute(
            "SELECT user_id, timestamp, position, role, message FROM messages WHERE id == ?",
            (message_id,),
        )

        if result := query.fetchone():
            user_id, timestamp, position, role, message = result
            return Message(
                message_id,
                user_id,
                datetime.fromtimestamp(timestamp),
                position,
                role,
                message,
            )
        return None

    @_requires_open_db
    def get_user_messages(self, user_id: int) -> list[Message] | None:
        query = self.db.execute(
            "SELECT id, timestamp, position, role, message FROM messages WHERE user_id == ? ORDER BY position ASC",
            (user_id,),
        )

        messages = []
        for result in query.fetchall():
            id, timestamp, position, role, message = result
            messages.append(
                Message(
                    id,
                    user_id,
                    datetime.fromtimestamp(timestamp),
                    position,
                    role,
                    message,
                )
            )
        return messages if len(messages) > 0 else None

    @_requires_open_db
    def get_nth_user_message(self, user_id: int, position: int) -> Message | None:
        query = self.db.execute(
            "SELECT id FROM messages WHERE user_id == ? AND position == ?",
            (user_id, position),
        )
        if result := query.fetchone():
            return self.get_message(result[0])  # type: ignore
        return None

    @_requires_open_db
    def add_message(
        self,
        user_id: int,
        timestamp: datetime | None,
        role: str,
        message: str,
        create_user_if_not_found: bool = True,
    ) -> None:
        if self.get_user(user_id) is None:
            if create_user_if_not_found and not self.add_user(user_id):
                raise CouldNotCreateUser(user_id)
            else:
                raise UserDoesNotExist(user_id)

        if timestamp is None:
            timestamp = datetime.now()

        next_message_position = self._next_user_message_position(user_id)

        with self.db as db:
            db.execute(
                "INSERT INTO messages(user_id, timestamp, position, role, message) VALUES (?, ?, ?, ?, ?)",
                (user_id, timestamp, next_message_position, role, message),
            )

    @_requires_open_db
    def delete_message(self, message: Message) -> None:
        # message position must be updated when a message is deleted
        messages_to_update = self._get_user_messages_ids_and_position_end_slice(message.user_id, message.position)
        update_data = [{"message_id": id, "new_position": position + 1} for id, position in messages_to_update]

        with self.db as db:
            db.executemany(
                "UPDATE messages SET position = :new_position WHERE id == :message_id",
                update_data,
            )
            db.execute("DELETE FROM messages WHERE id == ?", (message.id,))

    @_requires_open_db
    def delete_message_by_id(self, message_id: int) -> bool:
        """Returns `True` if message was removed, `False` if not found"""
        if message_to_delete := self.get_message(message_id):
            self.delete_message(message_to_delete)
            return True
        return False

    @_requires_open_db
    def delete_user_message_by_position(self, user_id: int, position: int) -> bool:
        """Returns `True` if message was removed, `False` if not found"""
        if message_to_delete := self.get_nth_user_message(user_id, position):
            self.delete_message(message_to_delete)
            return True
        return False

    @_requires_open_db
    def clear_user_messages(self, user_id: int) -> None:
        with self.db as db:
            db.execute("DELETE FROM messages WHERE user_id == ?", (user_id,))

    @_requires_open_db
    def _initialize_database(self) -> None:
        with self.db as db:
            db.execute("PRAGMA foreign_keys = ON;")

            db.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    system_prompt TEXT NOT NULL)"""
            )

            db.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE CASCADE,
                timestamp INTEGER NOT NULL,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL)"""
            )

    @_requires_open_db
    def _next_user_message_position(self, user_id: int) -> int:
        query = self.db.execute(
            "SELECT position FROM messages WHERE user_id == ? ORDER BY position DESC LIMIT 1",
            (user_id,),
        )

        if result := query.fetchone():
            return int(result[0]) + 1
        return 0

    @_requires_open_db
    def _get_user_messages_ids_and_position_end_slice(self, user_id: int, from_position: int) -> list[tuple[int, int]]:
        query = self.db.execute(
            "SELECT id, position FROM messages WHERE user_id == ? AND position > ? ORDER BY position ASC",
            (user_id, from_position),
        )

        results = []
        for result in query.fetchall():
            id, position = result
            results.append((id, position))
        return results
