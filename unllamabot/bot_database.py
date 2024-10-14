"""Utilities for interacting with database containing user conversations and preferences"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Callable


class DatabaseAlreadyOpen(Exception):
    pass


class DatabaseNotOpen(Exception):
    pass


class ParameterSetError(Exception):
    def __init__(self, description: str, *args: object) -> None:
        self.description = description
        super().__init__(*args)

    def __str__(self) -> str:
        return self.description


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
    temperature: float | None
    dynatemp_range: float | None
    dynatemp_exponent: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None
    n_predict: int | None
    n_keep: int | None
    tfs_z: float | None
    typical_p: float | None
    repeat_penalty: float | None
    repeat_last_n: int | None
    penalize_nl: bool | None
    presence_penalty: float | None
    frequency_penalty: float | None
    mirostat: int | None
    mirostat_tau: float | None
    mirostat_eta: float | None
    seed: int | None
    samplers: list[str] | None


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    BOT = "assistant"


@dataclass
class Message:
    id: int
    user_id: int
    timestamp: datetime
    position: int
    role: ChatRole
    message: str


def _adapt_datetime_iso(val: datetime) -> str:
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()


def _convert_datetime(val: bytes) -> datetime:
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.fromisoformat(val.decode())


sqlite3.register_adapter(datetime, _adapt_datetime_iso)
sqlite3.register_converter("datetime", _convert_datetime)


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
        query = self.db.execute(
            """SELECT
            system_prompt,
            temperature,
            dynatemp_range,
            dynatemp_exponent,
            top_k,
            top_p,
            min_p,
            n_predict,
            n_keep,
            tfs_z,
            typical_p,
            repeat_penalty,
            repeat_last_n,
            penalize_nl,
            presence_penalty,
            frequency_penalty,
            mirostat,
            mirostat_tau,
            mirostat_eta,
            seed,
            samplers
            FROM users WHERE id == ?""",
            (user_id,),
        )
        if query_result := query.fetchone():
            (
                system_prompt,
                temperature,
                dynatemp_range,
                dynatemp_exponent,
                top_k,
                top_p,
                min_p,
                n_predict,
                n_keep,
                tfs_z,
                typical_p,
                repeat_penalty,
                repeat_last_n,
                penalize_nl,
                presence_penalty,
                frequency_penalty,
                mirostat,
                mirostat_tau,
                mirostat_eta,
                seed,
                samplers,
            ) = query_result
            return User(
                user_id,
                system_prompt,
                temperature,
                dynatemp_range,
                dynatemp_exponent,
                top_k,
                top_p,
                min_p,
                n_predict,
                n_keep,
                tfs_z,
                typical_p,
                repeat_penalty,
                repeat_last_n,
                penalize_nl == 1,
                presence_penalty,
                frequency_penalty,
                mirostat,
                mirostat_tau,
                mirostat_eta,
                seed,
                samplers,
            )
        return None

    @_requires_open_db
    def get_or_create_user(self, user_id: int, system_prompt: str | None = None) -> User:
        """Returns an user. If it doesn't exist, it's created with provided configuration."""
        if not self.user_exists(user_id):
            self.add_user(user_id, system_prompt)
        return self.get_user(user_id)  # type: ignore

    @_requires_open_db
    def add_user(self, user_id: int, system_prompt: str | None = None) -> bool:
        """Returns True if adding was successful, False if user already exists.
        If system prompt is not provided, default one will be used."""
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        with self.db as db:
            try:
                query = db.execute(
                    "INSERT INTO users(id, system_prompt) VALUES (?, ?)",
                    (user_id, system_prompt),
                )
            except sqlite3.IntegrityError:
                logging.error(f"Tried to add an user with existing ID: {user_id}")
                return False
            except Exception as e:
                logging.critical(f"Unhandled sqlite exception raised in add_user: {e}!")
                raise
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
            db.execute(
                "UPDATE messages SET message = ? WHERE user_id == ? AND role == ?",
                (new_system_prompt, user_id, str(ChatRole.SYSTEM)),
            )

    def _set_user_gen_param(
        self, user_id: int, parameter_name: str, parameter_raw_value: str, parameter_type: type
    ) -> tuple[str, str]:
        """Helper function to set a user generation parameter."""
        query = self.db.execute(f"SELECT {parameter_name} FROM users WHERE id == ?", (user_id,))
        old_raw_value: str = query.fetchone()[0]

        try:
            new_param_value = parameter_type(parameter_raw_value)
            # fallback to `int`, because sqlite doesn't have `bool` type
            if parameter_type is bool:
                new_param_value = int(new_param_value)

            with self.db as db:
                db.execute(f"UPDATE users SET {parameter_name} = ? WHERE id == ?", (new_param_value, user_id))
        except ValueError:
            raise ParameterSetError(f"Invalid {parameter_name}: {parameter_raw_value}")

        query = self.db.execute(f"SELECT {parameter_name} FROM users WHERE id == ?", (user_id,))
        new_raw_value: str = query.fetchone()[0]

        return old_raw_value, new_raw_value

    @_requires_open_db
    def set_user_generation_parameter(
        self, user_id: int, parameter_name: str, parameter_raw_value: str, create_user_if_not_found: bool = True
    ) -> tuple[str, str]:
        """Parses and sets a parameter value.
        If parameter is set correctly, returns tuple containing it's `(old, new)` values as human-readable strings.
        If arguments are invalid, raises `ParameterSetError`.
        If user does not exists and `create_user_if_not_found` is `False`, raises `UserDoesNotExist`."""
        if not self.user_exists(user_id):
            if create_user_if_not_found:
                self.add_user(user_id)
            else:
                raise UserDoesNotExist(user_id)

        match parameter_name:
            case "temperature":
                return self._set_user_gen_param(user_id, "temperature", parameter_raw_value, float)
            case "dynatemp_range":
                return self._set_user_gen_param(user_id, "dynatemp_range", parameter_raw_value, float)
            case "dynatemp_exponent":
                return self._set_user_gen_param(user_id, "dynatemp_exponent", parameter_raw_value, float)
            case "top_k":
                return self._set_user_gen_param(user_id, "top_k", parameter_raw_value, int)
            case "top_p":
                return self._set_user_gen_param(user_id, "top_p", parameter_raw_value, float)
            case "min_p":
                return self._set_user_gen_param(user_id, "min_p", parameter_raw_value, float)
            case "n_predict":
                return self._set_user_gen_param(user_id, "n_predict", parameter_raw_value, int)
            case "n_keep":
                return self._set_user_gen_param(user_id, "n_keep", parameter_raw_value, int)
            case "tfs_z":
                return self._set_user_gen_param(user_id, "tfs_z", parameter_raw_value, float)
            case "typical_p":
                return self._set_user_gen_param(user_id, "typical_p", parameter_raw_value, float)
            case "repeat_penalty":
                return self._set_user_gen_param(user_id, "repeat_penalty", parameter_raw_value, float)
            case "repeat_last_n":
                return self._set_user_gen_param(user_id, "repeat_last_n", parameter_raw_value, int)
            case "penalize_nl":
                return self._set_user_gen_param(user_id, "penalize_nl", parameter_raw_value, bool)
            case "presence_penalty":
                return self._set_user_gen_param(user_id, "presence_penalty", parameter_raw_value, float)
            case "frequency_penalty":
                return self._set_user_gen_param(user_id, "frequency_penalty", parameter_raw_value, float)
            case "mirostat":
                return self._set_user_gen_param(user_id, "mirostat", parameter_raw_value, int)
            case "mirostat_tau":
                return self._set_user_gen_param(user_id, "mirostat_tau", parameter_raw_value, float)
            case "mirostat_eta":
                return self._set_user_gen_param(user_id, "mirostat_eta", parameter_raw_value, float)
            case "seed":
                return self._set_user_gen_param(user_id, "seed", parameter_raw_value, int)
            case "samplers":
                raise ParameterSetError("Samplers order configuration is currently WIP")
            case _:
                raise ParameterSetError(f"Unknown parameter: {parameter_name}")

    @_requires_open_db
    def user_has_messages(self, user_id: int) -> bool:
        query = self.db.execute("SELECT EXISTS(SELECT 1 FROM messages WHERE user_id == ?)", (user_id,))
        return bool(query.fetchone()[0] == 1)

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
                datetime.fromisoformat(timestamp),
                position,
                ChatRole(role),
                message,
            )
        return None

    @_requires_open_db
    def get_user_messages(self, user_id: int) -> list[Message]:
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
                    datetime.fromisoformat(timestamp),
                    position,
                    ChatRole(role),
                    message,
                )
            )
        return messages

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
        role: ChatRole,
        message: str,
        timestamp: datetime | None = None,
        create_user_if_not_found: bool = True,
    ) -> None:
        if not self.user_exists(user_id):
            if create_user_if_not_found:
                if not self.add_user(user_id):
                    raise CouldNotCreateUser(user_id)
            else:
                raise UserDoesNotExist(user_id)

        if timestamp is None:
            timestamp = datetime.now()

        next_message_position = self._next_user_message_position(user_id)

        with self.db as db:
            db.execute(
                "INSERT INTO messages(user_id, timestamp, position, role, message) VALUES (?, ?, ?, ?, ?)",
                (user_id, timestamp, next_message_position, str(role), message),
            )

    @_requires_open_db
    def delete_message(self, message: Message) -> None:
        # message position must be updated when a message is deleted
        messages_to_update = self._get_user_messages_ids_and_position_from(message.user_id, message.position)
        update_data = [{"message_id": id, "new_position": position - 1} for id, position in messages_to_update]

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
                    system_prompt TEXT NOT NULL,
                    temperature REAL,
                    dynatemp_range REAL,
                    dynatemp_exponent REAL,
                    top_k INTEGER,
                    top_p REAL,
                    min_p REAL,
                    n_predict INTEGER,
                    n_keep INTEGER,
                    tfs_z REAL,
                    typical_p REAL,
                    repeat_penalty REAL,
                    repeat_last_n INTEGER,
                    penalize_nl INTEGER,
                    presence_penalty REAL,
                    frequency_penalty REAL,
                    mirostat INTEGER,
                    mirostat_tau REAL,
                    mirostat_eta REAL,
                    seed INTEGER,
                    samplers TEXT)"""
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
    def _get_user_messages_ids_and_position_from(self, user_id: int, from_position: int) -> list[tuple[int, int]]:
        query = self.db.execute(
            "SELECT id, position FROM messages WHERE user_id == ? AND position > ? ORDER BY position ASC",
            (user_id, from_position),
        )

        results = []
        for result in query.fetchall():
            id, position = result
            results.append((id, position))
        return results
