"""Bot database unit tests"""

from datetime import datetime
from typing import Sequence
import pytest
from unllamabot.bot_database import (
    BotDatabase,
    DatabaseNotOpen,
    UserDoesNotExist,
)

# change to empty string if temp database is preferred
TEST_DB_PATH = ":memory:"
TEST_SYSTEM_PROMPT_DEFAULT = "This is a default system prompt"
TEST_SYSTEM_PROMPT_A = "This is a test system prompt!"
TEST_SYSTEM_PROMPT_B = "This is another, different system prompt."


def create_users(db: BotDatabase, users: Sequence[tuple[int, str | None]]) -> None:
    for id, prompt in users:
        assert db.user_exists(id) is False
        assert db.add_user(id, prompt) is True
        assert db.user_exists(id) is True


def validate_users(db: BotDatabase, users: Sequence[tuple[int, str | None]], system_prompt: str) -> None:
    for id, prompt in users:
        user = db.get_user(id)
        assert user is not None
        assert user.id == id
        assert user.system_prompt == (prompt if prompt is not None else system_prompt)


def add_messages(db: BotDatabase, messages: Sequence[tuple[int, str, str]], create_user: bool = False) -> None:
    for user_id, role, message in messages:
        db.add_message(user_id, None, role, message, create_user)


def validate_messages(db: BotDatabase, messages: Sequence[tuple[int, int, str, str]]) -> None:
    for user_id, position, role, message_text in messages:
        message = db.get_nth_user_message(user_id, position)
        assert message is not None
        assert message.user_id == user_id
        assert message.position == position
        assert message.role == role
        assert message.message == message_text


def test_creating_empty_database() -> None:
    db = BotDatabase()
    assert not db.is_open
    assert db.default_system_prompt == ""
    with pytest.raises(DatabaseNotOpen):
        db.close()

    db.open(TEST_DB_PATH)
    assert db.is_open
    db.close()


def test_adding_users() -> None:
    db = BotDatabase(TEST_DB_PATH)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, TEST_SYSTEM_PROMPT_A),
        (456, TEST_SYSTEM_PROMPT_B),
    )

    create_users(db, test_users_data)
    validate_users(db, test_users_data, "")


def test_adding_users_with_custom_default_prompt() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, TEST_SYSTEM_PROMPT_A),
        (456, None),
    )

    create_users(db, test_users_data)
    validate_users(db, test_users_data, TEST_SYSTEM_PROMPT_DEFAULT)


def test_changing_global_default_system_prompt() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, None),
        (456, TEST_SYSTEM_PROMPT_B),
    )

    create_users(db, test_users_data)
    validate_users(db, test_users_data, TEST_SYSTEM_PROMPT_DEFAULT)
    db.change_global_default_system_prompt(TEST_SYSTEM_PROMPT_A)
    validate_users(db, test_users_data, TEST_SYSTEM_PROMPT_A)


def test_changing_user_system_prompt() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, None),
        (456, TEST_SYSTEM_PROMPT_B),
    )

    create_users(db, test_users_data)
    validate_users(db, test_users_data, TEST_SYSTEM_PROMPT_DEFAULT)
    db.change_user_system_prompt(1, TEST_SYSTEM_PROMPT_A)
    with pytest.raises(UserDoesNotExist):
        db.change_user_system_prompt(555, TEST_SYSTEM_PROMPT_A, create_user_if_not_found=False)
    db.change_user_system_prompt(666, TEST_SYSTEM_PROMPT_A, create_user_if_not_found=True)

    expected_test_users_data = (
        (1, TEST_SYSTEM_PROMPT_A),
        (2, ""),
        (123, None),
        (456, TEST_SYSTEM_PROMPT_B),
        (666, TEST_SYSTEM_PROMPT_A),
    )
    validate_users(db, expected_test_users_data, TEST_SYSTEM_PROMPT_DEFAULT)


def test_deleting_users() -> None:
    db = BotDatabase(TEST_DB_PATH)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, TEST_SYSTEM_PROMPT_A),
        (456, TEST_SYSTEM_PROMPT_B),
    )
    test_messages = (
        (1, "user", "user message 1"),
        (1, "assistant", "assistant reply 1"),
        (1, "user", "user message 2"),
        (1, "assistant", "assistant reply 2"),
    )

    create_users(db, test_users_data)
    add_messages(db, test_messages)

    assert db.delete_user(1) is True
    assert db.delete_user(123) is True

    expected_test_users_data = (
        (2, ""),
        (456, TEST_SYSTEM_PROMPT_B),
    )
    validate_users(db, expected_test_users_data, "")
    assert len(db.get_user_messages(1)) == 0


def test_adding_message() -> None:
    db = BotDatabase(TEST_DB_PATH)
    create_users(db, ((1, None),))

    expected_timestamp = datetime(year=2221, month=11, day=11)
    db.add_message(
        1,
        expected_timestamp,
        "test_role",
        "test_message_content",
        create_user_if_not_found=False,
    )
    created_message = db.get_nth_user_message(1, 0)
    assert created_message is not None
    assert created_message.user_id == 1
    assert created_message.timestamp == expected_timestamp
    assert created_message.position == 0
    assert created_message.role == "test_role"
    assert created_message.message == "test_message_content"


def test_adding_message_and_creating_user() -> None:
    db = BotDatabase(TEST_DB_PATH)
    expected_timestamp = datetime(year=2222, month=11, day=22)

    assert not db.user_exists(123)
    db.add_message(
        123,
        expected_timestamp,
        "test_role",
        "test_message_content",
        create_user_if_not_found=True,
    )

    assert db.user_exists(123)
    created_message = db.get_nth_user_message(123, 0)
    assert created_message is not None
    assert created_message.user_id == 123
    assert created_message.timestamp == expected_timestamp
    assert created_message.position == 0
    assert created_message.role == "test_role"
    assert created_message.message == "test_message_content"


def test_adding_message_with_default_timestamp() -> None:
    db = BotDatabase(TEST_DB_PATH)

    expected_timestamp = datetime.now()
    db.add_message(123, None, "test_role", "test_content", create_user_if_not_found=True)
    created_message = db.get_nth_user_message(123, 0)

    # 100us of tolerance for CI
    assert (created_message.timestamp - expected_timestamp).microseconds <= 100


def test_adding_message_to_nonexistent_user_is_failing() -> None:
    db = BotDatabase(TEST_DB_PATH)

    expected_timestamp = datetime(year=2221, month=11, day=11)
    with pytest.raises(UserDoesNotExist):
        db.add_message(
            1,
            expected_timestamp,
            "test_role",
            "test_message_content",
            create_user_if_not_found=False,
        )


def test_getting_user_messages() -> None:
    db = BotDatabase(TEST_DB_PATH)
    test_messages = (
        (1, "user 1", "user message 1"),
        (1, "assistant", "assistant reply 1"),
        (1, "user 1", "user message 2"),
        (1, "assistant", "assistant reply 2"),
        (2, "user 2", "user message 1"),
        (2, "assistant", "assistant reply 1"),
        (2, "user 2", "user message 2"),
        (2, "assistant", "assistant reply 2"),
    )

    expected_user_messages = (
        (2, "user 2", "user message 1"),
        (2, "assistant", "assistant reply 1"),
        (2, "user 2", "user message 2"),
        (2, "assistant", "assistant reply 2"),
    )

    add_messages(db, test_messages, True)
    user_messages = db.get_user_messages(2)
    assert len(user_messages) == len(expected_user_messages)
    for (user_id, role, message_text), message in zip(expected_user_messages, user_messages):
        assert message.user_id == user_id
        assert message.role == role
        assert message.message == message_text

    nonexistent_user_messages = db.get_user_messages(555)
    assert len(nonexistent_user_messages) == 0


def test_getting_nth_user_message() -> None:
    pass


def test_deleting_messages() -> None:
    pass


def test_deleting_messages_by_id() -> None:
    pass


def test_deleting_user_messages_by_position() -> None:
    pass


def test_clearing_user_messages() -> None:
    db = BotDatabase(TEST_DB_PATH)
    test_messages = (
        (1, "user 1", "user message 1"),
        (1, "assistant", "assistant reply 1"),
        (1, "user 1", "user message 2"),
        (1, "assistant", "assistant reply 2"),
        (2, "user 2", "user message 1"),
        (2, "assistant", "assistant reply 1"),
        (2, "user 2", "user message 2"),
        (2, "assistant", "assistant reply 2"),
    )

    add_messages(db, test_messages, True)
    assert len(db.get_user_messages(1)) == 4
    assert len(db.get_user_messages(2)) == 4
    db.clear_user_messages(1)
    assert len(db.get_user_messages(1)) == 0
    assert len(db.get_user_messages(2)) == 4
