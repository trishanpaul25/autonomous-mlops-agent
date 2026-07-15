from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    dataset_id: str | None = None


class ChatResponse(BaseModel):
    user_prompt: str
    assistant_message: str | None = None

    run_id : str
    status: str
    execution_time: float | None = None
    completed_steps: list[str]
    logs: list[str]

    dataset_name: str | None = None
    rows: int | None = None
    columns: int | None = None

    problem_type: str | None = None
    target_column: str | None = None