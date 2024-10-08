"""Settings for a GridWorks JournalKeeper, readable from environment and/or from env files."""

from pydantic import ConfigDict, SecretStr
from pydantic_settings import BaseSettings

DEFAULT_ENV_FILE = ".env"


class Settings(BaseSettings):
    db_url: SecretStr = SecretStr(
        "postgresql://persister:PASSWD@journaldb.electricity.works/journaldb"
    )
    db_pass: SecretStr = SecretStr("Passwd")
    ops_genie_api_key: SecretStr = SecretStr("OpsGenieAPIKey")
    thermostat_api_key: SecretStr = SecretStr("ThermostatAPIKey")

    model_config = ConfigDict(
        env_prefix="gjk_",
        env_nested_delimiter="__",
        extra="ignore",
    )