import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger import logger

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[1] / ".env"
)

class LLMService:
    @staticmethod
    def get_llm():
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set in .env file.")
        model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")
        logger.info(f"Using LLM model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=float(os.getenv("TEMPERATURE", "0")),
        )

    @staticmethod
    def get_structured_llm(schema):
        return LLMService.get_llm().with_structured_output(schema)