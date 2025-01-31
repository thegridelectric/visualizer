from pydantic import ConfigDict, SecretStr
from pydantic_settings import BaseSettings

DEFAULT_ENV_FILE = ".env"

class Settings(BaseSettings):
    db_url: SecretStr = SecretStr(
        "postgresql://persister:PASSWD@journaldb.electricity.works/journaldb"
    )
    db_pass: SecretStr = SecretStr("Passwd")
    ops_genie_api_key: SecretStr = SecretStr("OpsGenieAPIKey")
    visualizer_api_password: SecretStr = SecretStr("ThermostatAPIKey")
    oak_owner_password: SecretStr = SecretStr("")

    model_config = ConfigDict(
        env_prefix="gjk_",
        env_nested_delimiter="__",
        extra="ignore",
    )