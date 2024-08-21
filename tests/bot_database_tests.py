"""Bot database unit tests"""

import pytest
from unllamabot.bot_database import BotDatabase, DatabaseNotOpen, UserDoesNotExist

# change to empty string if temp database is preferred
TEST_DB_PATH = ":memory:"
TEST_SYSTEM_PROMPT_DEFAULT = "This is a default system prompt"
TEST_SYSTEM_PROMPT_A = "This is a test system prompt!"
TEST_SYSTEM_PROMPT_B = "This is another, different system prompt."


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

    for id, prompt in test_users_data:
        assert db.user_exists(id) is False
        assert db.add_user(id, prompt) is True
        assert db.user_exists(id) is True
        user = db.get_user(id)
        assert user is not None
        assert user.id == id
        assert user.system_prompt == (prompt if prompt is not None else "")


def test_adding_users_with_custom_default_prompt() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, TEST_SYSTEM_PROMPT_A),
        (456, None),
    )

    for id, prompt in test_users_data:
        assert db.user_exists(id) is False
        assert db.add_user(id, prompt) is True
        assert db.user_exists(id) is True
        user = db.get_user(id)
        assert user is not None
        assert user.id == id
        assert user.system_prompt == (prompt if prompt is not None else TEST_SYSTEM_PROMPT_DEFAULT)


def test_changing_global_default_system_prompt() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, None),
        (456, TEST_SYSTEM_PROMPT_B),
    )

    for id, prompt in test_users_data:
        assert db.user_exists(id) is False
        assert db.add_user(id, prompt) is True
        assert db.user_exists(id) is True

    db.change_global_default_system_prompt(TEST_SYSTEM_PROMPT_A)

    for id, prompt in test_users_data:
        user = db.get_user(id)
        assert user is not None
        assert user.id == id
        assert user.system_prompt == (prompt if prompt is not None else TEST_SYSTEM_PROMPT_A)


def test_changing_user_system_prompt() -> None:
    db = BotDatabase(TEST_DB_PATH, TEST_SYSTEM_PROMPT_DEFAULT)
    test_users_data = (
        (1, None),
        (2, ""),
        (123, None),
        (456, TEST_SYSTEM_PROMPT_B),
    )

    for id, prompt in test_users_data:
        assert db.user_exists(id) is False
        assert db.add_user(id, prompt) is True
        assert db.user_exists(id) is True

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

    for id, prompt in expected_test_users_data:
        user = db.get_user(id)
        assert user is not None
        assert user.id == id
        assert user.system_prompt == (prompt if prompt is not None else TEST_SYSTEM_PROMPT_DEFAULT)
