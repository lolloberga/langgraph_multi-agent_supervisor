from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_MODEL_VERSION: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    STREAMING_MODEL: str = "false"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
