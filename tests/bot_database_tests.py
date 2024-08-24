"""Bot database unit tests"""

from datetime import datetime
from typing import Sequence
import pytest
from unllamabot.bot_database import (
    BotDatabase,
    DatabaseNotOpen,
    Message,
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


def test_functions_throw_on_unopened_database() -> None:
    db = BotDatabase()

    with pytest.raises(DatabaseNotOpen):
        db.close()

    with pytest.raises(DatabaseNotOpen):
        db.change_global_default_system_prompt("")

    with pytest.raises(DatabaseNotOpen):
        db.get_user(0)

    with pytest.raises(DatabaseNotOpen):
        db.add_user(0)

    with pytest.raises(DatabaseNotOpen):
        db.delete_user(0)

    with pytest.raises(DatabaseNotOpen):
        db.user_exists(0)

    with pytest.raises(DatabaseNotOpen):
        db.change_user_system_prompt(0, "")

    with pytest.raises(DatabaseNotOpen):
        db.get_message(0)

    with pytest.raises(DatabaseNotOpen):
        db.get_user_messages(0)

    with pytest.raises(DatabaseNotOpen):
        db.get_nth_user_message(0, 0)

    with pytest.raises(DatabaseNotOpen):
        db.add_message(0, None, "", "")

    with pytest.raises(DatabaseNotOpen):
        db.delete_message(Message(0, 0, datetime.now(), 0, "", ""))

    with pytest.raises(DatabaseNotOpen):
        db.delete_message_by_id(0)

    with pytest.raises(DatabaseNotOpen):
        db.delete_user_message_by_position(0, 0)

    with pytest.raises(DatabaseNotOpen):
        db.clear_user_messages(0)


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

    # 100ms of tolerance for CI
    assert abs(created_message.timestamp - expected_timestamp).microseconds <= 100_000


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
    db = BotDatabase(TEST_DB_PATH)
    test_messages = (
        (1, "user 1", "user message 1"),
        (1, "assistant", "assistant reply 1"),
        (1, "user 1", "user message 2"),  # this message should be fetched
        (1, "assistant", "assistant reply 2"),
        (2, "user 2", "user message 1"),
        (2, "assistant", "assistant reply 1"),  # this message should be fetched
        (2, "user 2", "user message 2"),
        (2, "assistant", "assistant reply 2"),
    )
    add_messages(db, test_messages, True)

    expected_message_a = test_messages[2]
    expected_message_b = test_messages[5]
    user_a_message = db.get_nth_user_message(1, 2)
    user_b_message = db.get_nth_user_message(2, 1)

    expected_user_id, expected_role, expected_message = expected_message_a
    assert user_a_message.user_id == expected_user_id
    assert user_a_message.role == expected_role
    assert user_a_message.message == expected_message

    expected_user_id, expected_role, expected_message = expected_message_b
    assert user_b_message.user_id == expected_user_id
    assert user_b_message.role == expected_role
    assert user_b_message.message == expected_message


def test_deleting_messages() -> None:
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
    expected_user_messages_post = (
        (2, 0, "user 2", "user message 1"),
        (2, 1, "user 2", "user message 2"),
        (2, 2, "assistant", "assistant reply 2"),
    )
    add_messages(db, test_messages, True)

    user_messages_pre = db.get_user_messages(2)
    assert len(user_messages_pre) == 4

    db.delete_message(user_messages_pre[1])
    user_messages_post = db.get_user_messages(2)
    assert len(user_messages_post) == len(expected_user_messages_post)

    for (expected_user_id, expected_position, expected_role, expected_message), user_message in zip(
        expected_user_messages_post, user_messages_post
    ):
        assert user_message.user_id == expected_user_id
        assert user_message.position == expected_position
        assert user_message.role == expected_role
        assert user_message.message == expected_message

    # sanity check - verify other user's messages haven't been touched
    assert len(db.get_user_messages(1)) == 4


def test_deleting_messages_by_id() -> None:
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
    expected_user_messages_post = (
        (1, 0, "user 1", "user message 1"),
        (1, 1, "assistant", "assistant reply 1"),
        (1, 2, "assistant", "assistant reply 2"),
    )
    add_messages(db, test_messages, True)

    user_messages_pre = db.get_user_messages(1)
    assert len(user_messages_pre) == 4

    msg_to_delete = db.get_user_messages(1)[2]
    assert db.delete_message_by_id(msg_to_delete.id) is True
    assert db.delete_message_by_id(0xDEADBEEF) is False
    user_messages_post = db.get_user_messages(1)
    assert len(user_messages_post) == len(expected_user_messages_post)

    for (expected_user_id, expected_position, expected_role, expected_message), user_message in zip(
        expected_user_messages_post, user_messages_post
    ):
        assert user_message.user_id == expected_user_id
        assert user_message.position == expected_position
        assert user_message.role == expected_role
        assert user_message.message == expected_message

    # sanity check - verify other user's messages haven't been touched
    assert len(db.get_user_messages(2)) == 4


def test_deleting_user_messages_by_position() -> None:
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
    expected_user_messages_post = (
        (1, 0, "user 1", "user message 1"),
        (1, 1, "assistant", "assistant reply 2"),
    )
    add_messages(db, test_messages, True)

    user_messages_pre = db.get_user_messages(1)
    assert len(user_messages_pre) == 4

    assert db.delete_user_message_by_position(1, 1) is True
    assert db.delete_user_message_by_position(1, 1) is True
    assert db.delete_user_message_by_position(123, 2) is False
    assert db.delete_user_message_by_position(2, 10) is False
    user_messages_post = db.get_user_messages(1)
    assert len(user_messages_post) == len(expected_user_messages_post)

    for (expected_user_id, expected_position, expected_role, expected_message), user_message in zip(
        expected_user_messages_post, user_messages_post
    ):
        assert user_message.user_id == expected_user_id
        assert user_message.position == expected_position
        assert user_message.role == expected_role
        assert user_message.message == expected_message

    # sanity check - verify other user's messages haven't been touched
    assert len(db.get_user_messages(2)) == 4


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
