from pydantic_settings import BaseSettings, SettingsConfigDict
from server.core.constants import FileType

class Settings(BaseSettings):

    APP_NAME: str = "Autonomous MLOps AI"

    APP_VERSION: str = "1.0.0"

    DEBUG: bool = True

    MAX_UPLOAD_SIZE: int = 100  # MB

    LOG_LEVEL: str = "INFO"

    #database
    DATABASE_URL: str

    #auth
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    ALLOWED_EXTENSIONS: list[FileType] = [
        FileType.CSV,
        FileType.EXCEL,
        FileType.JSON,
        FileType.ZIP
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )



settings = Settings()