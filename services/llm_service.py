import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[1] / ".env"
)


class LLMService:
    @staticmethod
    def get_llm():
        return ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME", "gemini-1.5-flash"),
            temperature=float(os.getenv("TEMPERATURE", "0")),
        )

    @staticmethod
    def get_structured_llm(schema):
        return LLMService.get_llm().with_structured_output(schema)