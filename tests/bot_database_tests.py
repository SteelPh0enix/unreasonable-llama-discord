"""Bot database unit tests"""

from datetime import datetime
from typing import Sequence
import pytest
from unllamabot.bot_database import (
    BotDatabase,
    ChatRole,
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


def add_messages(db: BotDatabase, messages: Sequence[tuple[int, ChatRole, str]], create_user: bool = False) -> None:
    for user_id, role, message in messages:
        db.add_message(user_id, role, message, create_user_if_not_found=create_user)


def validate_messages_ignoring_timestamps(db: BotDatabase, messages: Sequence[tuple[int, int, ChatRole, str]]) -> None:
    for user_id, position, role, message_text in messages:
        message = db.get_nth_user_message(user_id, position)
        assert message is not None
        assert message.user_id == user_id
        assert message.position == position
        assert message.role == role
        assert message.message == message_text


def validate_messages(db: BotDatabase, messages: Sequence[tuple[int, datetime, int, ChatRole, str]]) -> None:
    for user_id, timestamp, position, role, message_text in messages:
        message = db.get_nth_user_message(user_id, position)
        assert message is not None
        assert message.user_id == user_id
        assert message.timestamp == timestamp
        assert message.position == position
        assert message.role == role
        assert message.message == message_text


# --------------------------------------------------------------------------------------------------


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
        db.get_or_create_user(0)

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
        db.add_message(0, "", "")

    with pytest.raises(DatabaseNotOpen):
        db.delete_message(Message(0, 0, datetime.now(), 0, ChatRole.SYSTEM, ""))

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

    # check if user cannot be added with same ID again
    assert db.add_user(1) is False
    assert db.add_user(1, "custom prompt") is False


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


def test_get_or_create_user() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    db.add_user(1, "Custom system prompt")
    db.add_user(2, "Different custom system prompt")

    user_existing_a = db.get_or_create_user(1)
    user_existing_b = db.get_or_create_user(2, "This should be ignored")
    user_created = db.get_or_create_user(3)
    user_created_custom_prompt = db.get_or_create_user(4, "Another custom system prompt")

    assert user_existing_a.id == 1
    assert user_existing_a.system_prompt == "Custom system prompt"
    assert user_existing_b.id == 2
    assert user_existing_b.system_prompt == "Different custom system prompt"
    assert user_created.id == 3
    assert user_created.system_prompt == TEST_SYSTEM_PROMPT_DEFAULT
    assert user_created_custom_prompt.id == 4
    assert user_created_custom_prompt.system_prompt == "Another custom system prompt"


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
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),
        (1, ChatRole.BOT, "assistant reply 2"),
    )

    create_users(db, test_users_data)
    add_messages(db, test_messages)

    assert db.delete_user(1) is True
    assert db.delete_user(123) is True
    assert db.delete_user(9999) is False
    assert db.delete_user(123) is False

    expected_test_users_data = (
        (2, ""),
        (456, TEST_SYSTEM_PROMPT_B),
    )
    validate_users(db, expected_test_users_data, "")
    assert len(db.get_user_messages(1)) == 0


def test_adding_messages() -> None:
    db = BotDatabase(TEST_DB_PATH)
    create_users(db, ((1, None),))
    expected_timestamp_a = datetime(year=2222, month=11, day=22)
    expected_timestamp_b = datetime(year=2223, month=11, day=22)
    expected_messages = (
        (123, expected_timestamp_a, 0, ChatRole.USER, "user request"),
        (123, expected_timestamp_b, 1, ChatRole.BOT, "assistant answer"),
    )
    for user_id, timestamp, _, role, content in expected_messages:
        db.add_message(user_id, role, content, timestamp=timestamp, create_user_if_not_found=True)
    validate_messages(db, expected_messages)


def test_adding_message_and_creating_user() -> None:
    db = BotDatabase(TEST_DB_PATH)
    expected_timestamp_a = datetime(year=2222, month=11, day=22)
    expected_timestamp_b = datetime(year=2223, month=11, day=22)
    expected_messages = (
        (123, expected_timestamp_a, 0, ChatRole.USER, "user request"),
        (123, expected_timestamp_b, 1, ChatRole.BOT, "assistant answer"),
    )

    assert not db.user_exists(123)
    for user_id, timestamp, _, role, content in expected_messages:
        db.add_message(user_id, role, content, timestamp=timestamp, create_user_if_not_found=True)
    assert db.user_exists(123)
    validate_messages(db, expected_messages)


def test_adding_message_with_default_timestamp() -> None:
    db = BotDatabase(TEST_DB_PATH)

    expected_timestamp = datetime.now()
    db.add_message(123, ChatRole.SYSTEM, "test_content", create_user_if_not_found=True)
    created_message = db.get_nth_user_message(123, 0)

    # 100ms of tolerance for CI
    assert abs(created_message.timestamp - expected_timestamp).microseconds <= 100_000


def test_adding_message_to_nonexistent_user() -> None:
    db = BotDatabase(TEST_DB_PATH)

    expected_timestamp = datetime(year=2221, month=11, day=11)
    with pytest.raises(UserDoesNotExist):
        db.add_message(
            1,
            ChatRole.SYSTEM,
            "test_message_content",
            timestamp=expected_timestamp,
            create_user_if_not_found=False,
        )


def test_getting_user_messages() -> None:
    db = BotDatabase(TEST_DB_PATH)
    test_messages = (
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),
        (1, ChatRole.BOT, "assistant reply 2"),
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
    )

    expected_user_messages = (
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
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
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),  # this message should be fetched
        (1, ChatRole.BOT, "assistant reply 2"),
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),  # this message should be fetched
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
    )
    add_messages(db, test_messages, True)

    expected_message_a = test_messages[2]
    expected_message_b = test_messages[5]
    user_a_message = db.get_nth_user_message(1, 2)
    user_b_message = db.get_nth_user_message(2, 1)

    expected_user_id, expected_role, expected_message = expected_message_a
    assert user_a_message.user_id == expected_user_id
    assert user_a_message.position == 2
    assert user_a_message.role == expected_role
    assert user_a_message.message == expected_message

    expected_user_id, expected_role, expected_message = expected_message_b
    assert user_b_message.user_id == expected_user_id
    assert user_b_message.position == 1
    assert user_b_message.role == expected_role
    assert user_b_message.message == expected_message


def test_deleting_messages() -> None:
    db = BotDatabase(TEST_DB_PATH)
    test_messages = (
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),
        (1, ChatRole.BOT, "assistant reply 2"),
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
    )
    expected_user_messages_post = (
        (2, 0, ChatRole.USER, "user message 1"),
        (2, 1, ChatRole.USER, "user message 2"),
        (2, 2, ChatRole.BOT, "assistant reply 2"),
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
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),
        (1, ChatRole.BOT, "assistant reply 2"),
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
    )
    expected_user_messages_post = (
        (1, 0, ChatRole.USER, "user message 1"),
        (1, 1, ChatRole.BOT, "assistant reply 1"),
        (1, 2, ChatRole.BOT, "assistant reply 2"),
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
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),
        (1, ChatRole.BOT, "assistant reply 2"),
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
    )
    expected_user_messages_post = (
        (1, 0, ChatRole.USER, "user message 1"),
        (1, 1, ChatRole.BOT, "assistant reply 2"),
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
        (1, ChatRole.USER, "user message 1"),
        (1, ChatRole.BOT, "assistant reply 1"),
        (1, ChatRole.USER, "user message 2"),
        (1, ChatRole.BOT, "assistant reply 2"),
        (2, ChatRole.USER, "user message 1"),
        (2, ChatRole.BOT, "assistant reply 1"),
        (2, ChatRole.USER, "user message 2"),
        (2, ChatRole.BOT, "assistant reply 2"),
    )

    add_messages(db, test_messages, True)
    assert len(db.get_user_messages(1)) == 4
    assert len(db.get_user_messages(2)) == 4
    db.clear_user_messages(1)
    assert len(db.get_user_messages(1)) == 0
    assert len(db.get_user_messages(2)) == 4
    db.clear_user_messages(123)
    assert len(db.get_user_messages(1)) == 0
    assert len(db.get_user_messages(2)) == 4
