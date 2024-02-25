import os
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# some definitions
APP_NAME = "metaphor_political_speeches"
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = os.path.join(BASE_DIR, "src")


class Modus(str, Enum):
    """
        An enumeration of the modes a model can be in.

        MULTI_LABEL: The model is in multi-label classification mode.
        BINARY: The model is in binary classification mode.
    """
    MULTI_LABEL = "multi_label"
    BINARY = "binary"


class Settings(BaseSettings):
    """
        Environmental settings for the application are defined in this class.

        Represents application settings like database host, port,
        batch sizes, and various paths used in the application.
    """
    db_host: str = Field(default=None, description="host for mongodb database")
    db_port: int = Field(default=27017, description="port number")
    db_name: str = Field(default="political_speeches_local", description="collection name")
    metaphor_dir: str = Field(default="", description="path to metaphors directory")
    mwe_dir: str = Field(default="", description="path to mwes directory")
    mwe_test: str = Field(default="", description="path to mwes test directory")
    mwe_train: str = Field(default="", description="path to mwes train directory")
    mwe_val: str = Field(default="", description="path to mwes validation directory")
    batch_train: int = Field(default=16, description="number of batches in training")
    batch_test: int = Field(default=1, description="number of batches in testing")
    K: int = Field(default=10, description="number of folds")
    epochs: int = Field(default=5, description="number of epochs")
    num_total_steps: int = Field(default=500, description="number of total steps")
    num_warmup_steps: int = Field(default=100, description="number of warmup steps")
    max_len: int = Field(default=90, description="number of maximum sentence length")
    heads: int = Field(default=2, description="number of heads")
    heads_mwe: int = Field(default=4, description="number of mwe heads")
    dropout: float = Field(default=0.6, description="dropout value")
    language_model: str = Field(default="", description="name of language model")
    model: str = Field(default="", description="name of transformer model")
    model_dir: str = Field(default="", description="path to model directory")
    modus: Modus = Field(default=Modus.BINARY, description="classification mode")
    init_db: bool = Field(default=False, description="whether to initialize db or not")
    model_config = SettingsConfigDict(env_file=os.path.join(SRC_DIR, ".env"), protected_namespaces=())

    def mdb_connection_string(self):
        """
            Creates and returns a MongoDB connection string.

            @returns mongodb string
        """
        return f"mongodb://{self.db_host}:{self.db_port}"


def get_settings() -> Settings:
    """
        Retrieves and returns the settings.
    """
    return Settings()
