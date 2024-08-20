"""Bot database unit tests"""

import pytest
from unllamabot.bot_database import BotDatabase, DatabaseNotOpen

# change to empty string if temp database is preferred
TEST_DB_PATH = ":memory:"


def test_creating_empty_database() -> None:
    db = BotDatabase()
    assert not db.is_open
    with pytest.raises(DatabaseNotOpen):
        db.close()

    db.open(TEST_DB_PATH)
    assert db.is_open
    db.close()
