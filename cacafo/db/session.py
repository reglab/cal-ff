from urllib.parse import quote_plus

import rl.utils.io
import sqlalchemy as sa
from playhouse.postgres_ext import PostgresqlExtDatabase


def get_postgres_uri(
    postgres_host: str = rl.utils.io.getenv("POSTGRES_HOST"),
    postgres_port: str = rl.utils.io.getenv("POSTGRES_PORT"),
    postgres_user: str = rl.utils.io.getenv("POSTGRES_USER"),
    postgres_password: str = rl.utils.io.getenv("POSTGRES_PASSWORD"),
    postgres_db: str = rl.utils.io.getenv("POSTGRES_DB"),
):
    if any(
        [
            not postgres_host,
            not postgres_port,
            not postgres_user,
            not postgres_db,
        ]
    ):
        raise ValueError(
            "You must provide env variables POSTGRES_HOST, POSTGRES_PORT, "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB."
        )

    postgres_password = quote_plus(postgres_password or "")
    return (
        f"postgresql://{postgres_user}:{postgres_password}"
        f"@{postgres_host}:{postgres_port}/{postgres_db}"
    )


def get_peewee_connection(
    **kwargs,
):
    """
    Get a Peewee database object for the Postgres database.
    kwargs are passed to get_postgres_uri.
    """
    return PostgresqlExtDatabase(get_postgres_uri(**kwargs))


def get_sqlalchemy_engine(
    **kwargs,
):
    """
    Get a SQLAlchemy engine object for the Postgres database.
    kwargs are passed to get_postgres_uri.
    """
    return sa.create_engine(get_postgres_uri(**kwargs))


def get_sqlalchemy_session(
    **kwargs,
):
    """
    Get a SQLAlchemy session object for the Postgres database.
    kwargs are passed to get_sqlalchemy_engine.
    """
    return sa.orm.sessionmaker(bind=get_sqlalchemy_engine(**kwargs))()
