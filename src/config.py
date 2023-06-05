import os
from pathlib import Path

from pydantic import BaseSettings

APP_NAME = "metaphor_political_speeches"
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = os.path.join(BASE_DIR, "src")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

LOGGER_NAME = f"{APP_NAME}log"
LOG_FILE_NAME = f"{APP_NAME}.log"


class Settings(BaseSettings):
    db_host: str = None
    db_port: int = 27017
    db_name: str = "political_speeches_local"

    class Config:
        env_file = os.path.join(SRC_DIR, ".env")

    def mdb_connection_string(self):
        return f"mongodb://{self.db_host}:{self.db_port}"


def get_settings() -> Settings:
    return Settings()
