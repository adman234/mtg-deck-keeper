from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    return cfg


def upgrade() -> None:
    command.upgrade(alembic_config(), "head")


def revision(message: str) -> None:
    """Developer helper: python -m deck_keeper.migrate "message" """
    command.revision(alembic_config(), message=message, autogenerate=True)


if __name__ == "__main__":
    import sys

    revision(" ".join(sys.argv[1:]) or "change")
