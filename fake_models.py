from sqlalchemy import BigInteger, Column, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()

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