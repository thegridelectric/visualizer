import logging
from typing import List

from sqlalchemy import BigInteger, Column, String, UniqueConstraint, tuple_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base

# Define the base class
Base = declarative_base()

LOG_FORMAT = (
    "%(levelname) -10s %(asctime)s %(name) -30s %(funcName) "
    "-35s %(lineno) -5d: %(message)s"
)
LOGGER = logging.getLogger(__name__)


class MessageSql(Base):
    __tablename__ = "messages"
    message_id = Column(String, primary_key=True)
    from_alias = Column(String, nullable=False)
    message_type_name = Column(String, nullable=False)
    message_persisted_ms = Column(BigInteger, nullable=False)
    payload = Column(JSONB, nullable=False)
    message_created_ms = Column(BigInteger)

    __table_args__ = (
        UniqueConstraint(
            "from_alias",
            "message_type_name",
            "message_persisted_ms",
            name="uq_from_type_message",
        ),
    )

    def to_dict(self):
        d = {
            "MessageId": self.message_id,
            "FromAlias": self.from_alias,
            "MessageTypeName": self.message_type_name,
            "MessagePersistedMs": self.message_persisted_ms,
            "Payload": self.payload,
        }
        if self.message_created_ms:
            d["MessageCreatedMs"] = self.message_created_ms
        return d


def bulk_insert_messages(session: Session, message_list: List[MessageSql]):
    """
    Idempotently bulk inserts MessageSql into the journaldb messages table,
    inserting only those whose primary keys do not already exist AND that
    don't violate the from_alias, type_name, message_persisted_ms uniqueness
    constraint.

    Args:
        session (Session): An active SQLAlchemy session used for database operations.
        message_list (List[MessageSql]): A list of MessageSql objects to be conditionally
        inserted into the messages table of the journaldb database

    Returns:
        None
    """
    if not all(isinstance(obj, MessageSql) for obj in message_list):
        raise ValueError("All objects in message_list must be MessageSql objects")

    try:
        pk_column = MessageSql.message_id
        unique_columns = [
            MessageSql.from_alias,
            MessageSql.message_type_name,
            MessageSql.message_persisted_ms,
        ]

        pk_set = set()
        unique_set = set()

        for message in message_list:
            pk_set.add(message.message_id)
            unique_set.add(tuple(getattr(message, col.name) for col in unique_columns))

        existing_pks = set(session.query(pk_column).filter(pk_column.in_(pk_set)).all())

        existing_uniques = set(
            session.query(*unique_columns)
            .filter(tuple_(*unique_columns).in_(unique_set))
            .all()
        )

        new_messages = [
            msg
            for msg in message_list
            if msg.message_id not in existing_pks
            and tuple(getattr(msg, col.name) for col in unique_columns)
            not in existing_uniques
        ]
        print(f"Inserting {len(new_messages)} out of {len(message_list)}")
        session.bulk_save_objects(new_messages)
        session.commit()

    except NoSuchTableError as e:
        print(f"Error: The table does not exist. {e}")
        session.rollback()
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")
        session.rollback()