from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    user_prompt: str
    # assistant_message: str = ""   # We'll add this later

    status: str
    completed_steps: list[str]
    logs: list[str]

    dataset_name: str | None = None
    rows: int | None = None
    columns: int | None = None

    problem_type: str | None = None
    target_column: str | None = None